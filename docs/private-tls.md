# IP 기반 사설 인증서와 Uvicorn mTLS

MTPLX를 리버스 프록시 없이 Uvicorn에서 직접 HTTPS로 실행하는 방법입니다.
도메인은 필요 없습니다. 서버 인증서의 SAN(Subject Alternative Name)에 **실제로 접속할 IP**를 넣습니다.
클라이언트도 인증서를 제시하는 상호 TLS(mTLS)를 사용하며, 기존 API 키 인증을 함께 유지합니다.

## 기본 실행과 선택 옵션

옵션을 생략하면 기존 동작을 유지합니다. localhost는 HTTP로 실행할 수 있습니다.

```sh
mtplx serve --host 127.0.0.1 --port 8000
```

아래 옵션은 `mtplx serve`, `mtplx quickstart`, `python -m mtplx.server.openai`에서 지원합니다.

| 옵션 | 의미 |
| --- | --- |
| `--ssl-certfile PATH` | 서버 인증서/체인 PEM. 개인키와 함께 지정하면 HTTPS 활성화 |
| `--ssl-keyfile PATH` | 암호 없는 서버 개인키 PEM |
| `--ssl-ca-certs PATH` | 클라이언트 인증서 검증용 CA PEM |
| `--ssl-require-client-cert` | 인증서가 없거나 신뢰할 수 없는 클라이언트의 TLS 연결 거부 |
| `--log-privacy metadata-only` | 원문 로그·trace·capture 차단, 수치 통계와 캐시 유지 |

CA 옵션과 클라이언트 인증서 필수 옵션은 함께 지정합니다. 서버 인증서/개인키만 지정하면 단방향 HTTPS입니다.
잘못된 인증서 경로·키 조합은 모델 로딩 전에 실패합니다. HTTP로 자동 전환하지 않습니다.
HTTPS로 실행한 포트는 localhost에서 접속하더라도 HTTPS입니다. 같은 리스너에서 HTTP/HTTPS를 혼용하지 않습니다.

## 1. 사설 CA와 인증서 만들기

OpenSSL 3.x를 사용합니다. `openssl version`으로 확인하세요. macOS에서는 필요하면 Homebrew OpenSSL의 실행 경로를 사용합니다.
아래의 `192.168.1.50`을 서버의 고정 IP로 바꿉니다. `0.0.0.0`은 수신 주소이며 인증서의 접속 IP가 아닙니다.

인증서 발급 작업은 별도의 보호된 디렉터리에서 수행합니다. 기존 파일이 있는 디렉터리에서 명령을 다시 실행하지 마세요.

```sh
umask 077
mkdir mtplx-pki
cd mtplx-pki

# CA 개인키는 암호로 보호합니다. 발급 시 암호를 입력합니다.
openssl genpkey -algorithm RSA -aes-256-cbc \
  -pkeyopt rsa_keygen_bits:3072 -out ca.key
openssl req -x509 -new -sha256 -days 3650 \
  -key ca.key -out ca.crt -subj '/CN=MTPLX Private CA' \
  -addext 'basicConstraints=critical,CA:TRUE,pathlen:0' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign'

openssl req -new -newkey rsa:2048 -nodes \
  -keyout server.key -out server.csr -subj '/CN=MTPLX Server'
cat > server.ext <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=IP:192.168.1.50,IP:127.0.0.1,IP:::1
EOF
openssl x509 -req -sha256 -days 365 -in server.csr \
  -CA ca.crt -CAkey ca.key -CAcreateserial \
  -extfile server.ext -out server.crt

openssl req -new -newkey rsa:2048 -nodes \
  -keyout client.key -out client.csr -subj '/CN=MTPLX Pydantic AI Client'
cat > client.ext <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=clientAuth
EOF
openssl x509 -req -sha256 -days 365 -in client.csr \
  -CA ca.crt -CAkey ca.key -CAcreateserial \
  -extfile client.ext -out client.crt

chmod 600 ca.key server.key client.key
openssl verify -CAfile ca.crt -purpose sslserver -verify_ip 192.168.1.50 server.crt
openssl verify -CAfile ca.crt -purpose sslclient client.crt
```

- 서버에 배포: `server.crt`, `server.key`, `ca.crt`.
- 클라이언트에 배포: `client.crt`, `client.key`, `ca.crt`, API 키 파일.
- `ca.key`는 발급 장소에만 보관합니다. 서버·클라이언트에 배포하지 않습니다.
- CA 인증서는 신뢰할 수 있는 경로로 전달합니다. 클라이언트별로 별도 키/인증서를 발급할 수 있습니다.
- IP가 바뀌면 서버 인증서를 해당 SAN으로 재발급합니다. 만료 전에 갱신하고 서버를 재시작합니다.
- 이 구성에는 인증서별 폐기 목록 자동 조회가 없습니다. 발급된 클라이언트 인증서는 신뢰 CA와 유효기간 내에서 허용됩니다. 유출 시 API 키 및 필요한 CA/인증서를 교체하세요.

## 2. 서버 실행

인증서는 저장소 밖에 둡니다. 다음 경로는 예시입니다.

```sh
mtplx serve --host 0.0.0.0 --port 8443 \
  --api-key-file "$HOME/.mtplx/api-key" \
  --ssl-certfile "$HOME/.mtplx/tls/server.crt" \
  --ssl-keyfile "$HOME/.mtplx/tls/server.key" \
  --ssl-ca-certs "$HOME/.mtplx/tls/ca.crt" \
  --ssl-require-client-cert \
  --log-privacy metadata-only
```

`--api-key-file`은 기존 기능이며 파일이 없으면 생성합니다. API 키는 클라이언트에 안전하게 전달합니다.
모델 선택은 기존 `--model`, `--model-id` 옵션을 사용합니다. mTLS는 `/health`를 포함한 모든 경로에 적용됩니다.
브라우저/대시보드도 CA 신뢰 설정과 클라이언트 인증서가 필요합니다. 기존 HTTP 전용 관리·연결 명령이 mTLS 클라이언트로 자동 설정되지는 않습니다.

```sh
curl --cacert ca.crt --cert client.crt --key client.key \
  https://192.168.1.50:8443/health
```

인증서 없는 연결, 다른 CA의 클라이언트 인증서, SAN과 다른 서버 IP는 실패해야 합니다.
인증서 검증을 끄는 `-k`/`verify=False`는 사용하지 않습니다.

## 3. Pydantic AI 클라이언트

[실행 예제](../examples/pydantic-ai-mtls.py)는 `pydantic-ai-slim[openai]==2.44.0`, `httpx2==2.13.0`으로 검증합니다.
서버 기본 의존성에 Pydantic AI를 추가하지 않습니다. `uv run`은 예제의 별도 의존성을 설치합니다.

```sh
uv run examples/pydantic-ai-mtls.py \
  --base-url https://192.168.1.50:8443/v1 \
  --model YOUR_SERVED_MODEL_ID \
  --ca /path/to/ca.crt \
  --cert /path/to/client.crt \
  --key /path/to/client.key \
  --api-key-file /path/to/api-key
```

프롬프트는 숨김 입력으로 받습니다. 응답은 메모리의 `result.output`에 있으며 예제는 토큰 사용량만 출력합니다.
TLS 설정의 핵심은 다음과 같습니다.

```python
ctx = ssl.create_default_context(cafile="ca.crt")
ctx.load_cert_chain("client.crt", "client.key")
async with httpx2.AsyncClient(verify=ctx, trust_env=False) as client:
    provider = OpenAIProvider(
        base_url="https://192.168.1.50:8443/v1",
        api_key=api_key,
        http_client=client,
    )
    agent = Agent(OpenAIChatModel(model_id, provider=provider))
    agent.instrument = False
    result = await agent.run(prompt)
```

`ca.crt`는 서버 검증용이고 `client.crt`/`client.key`는 서버에 클라이언트 신원을 증명하는 용도입니다.
`trust_env=False`는 환경변수의 프록시를 거치지 않도록 합니다. 인증서 신뢰는 위 context로 명시합니다.
Pydantic AI v2는 구형 `httpx.AsyncClient`도 받지만 deprecated이므로 새 예제는 `httpx2`를 사용합니다.
애플리케이션에서 별도로 HTTP debug 로깅, Logfire/OpenTelemetry 본문 수집, 대화 파일 저장을 켜지 마세요. 서버 옵션은 클라이언트의 로그 설정을 제어하지 않습니다.

## 4. 통계 전용 로그의 범위

`--log-privacy metadata-only`는 TLS와 독립적이므로 localhost HTTP에서도 사용할 수 있습니다.

- 요청 JSONL에는 허용한 수치 필드만 저장합니다: 입력/출력 토큰 수, 처리 시간, TTFT, 처리 속도, 캐시 적중 수치 등.
- 기본 저장 경로는 `~/.mtplx/logs/request-log-<port>.jsonl`입니다. 기존 환경변수 `MTPLX_REQUEST_LOG_JSONL`로 경로를 변경하거나 `off`로 저장을 끌 수 있습니다.
- 프롬프트/응답, 사용자 미리보기와 해시, 도구 인자, token ID 배열, 사용자 지정 ID, 임의 오류 문자열을 요청 로그에서 제외합니다.
- flight recorder(오류·취소 시 생성 텍스트 포함), request capture, decode trace, stream census, postcommit mismatch 덤프, eval audit 파일을 차단합니다.
- `MTPLX_REQUEST_LOG_CONTENT=1`, capture/trace 환경변수나 flight recorder 옵션이 있어도 이 정책이 우선합니다.
- 실행 안내 이후 자유 형식 stdout/stderr를 억제합니다. 오류 로그는 본문과 traceback 대신 고정 경고 문구만 출력합니다. 통계는 JSONL과 API에서 확인합니다.
- API 응답과 메모리 내 대시보드 데이터는 정상 동작합니다. 이 옵션은 요청/응답 데이터를 앱에서 사용하지 못하게 하는 기능이 아닙니다.
- 기존 로그는 삭제하지 않습니다. 과거에 저장한 파일은 운영자가 별도로 관리해야 합니다.

**KV/SSD 캐시는 그대로 유지됩니다.** SSD 캐시에는 KV 텐서와 전체 token ID 배열(`token_ids_json`)이 저장됩니다.
해당 tokenizer로 텍스트 복원이 가능하므로 캐시는 암호화된 데이터로 간주할 수 없습니다.
이 옵션은 로그의 원문 미저장 정책이며 디스크 캐시 암호화, 메모리 보호, OS crash dump 제어를 제공하지 않습니다.

## 참고

- [Uvicorn HTTPS 설정](https://www.uvicorn.org/settings/#https)
- [OpenSSL SAN과 인증서 확장](https://docs.openssl.org/3.4/man5/x509v3_config/)
- [Pydantic AI OpenAI 호환 provider](https://pydantic.dev/docs/ai/models/openai/)
- [SSL context와 클라이언트 인증서](https://www.python-httpx.org/advanced/ssl/)
