import Foundation
import XCTest

@testable import MTPLXAppCore

final class ModelRemovalServiceTests: XCTestCase {
    func testRemovalResultDecodesCLIJSON() throws {
        let data = Data(
            #"{"repo_id":"owner/pack","path":"/cache/owner--pack","removed":true,"size_bytes_removed":42000000000}"#.utf8
        )

        let result = try JSONDecoder().decode(CachedModelRemovalResult.self, from: data)

        XCTAssertEqual(result.repoID, "owner/pack")
        XCTAssertEqual(result.path, "/cache/owner--pack")
        XCTAssertTrue(result.removed)
        XCTAssertEqual(result.sizeBytesRemoved, 42_000_000_000)
    }

    /// The app owns the confirmation, so the CLI must not ask again: without
    /// `--yes` it refuses a non-interactive delete and exits 1. The cache
    /// root is always explicit and the entry travels by folder name.
    func testRemovalArgumentsConfirmAndPinTheCacheRoot() {
        XCTAssertEqual(
            ModelDownloader.removalArguments(
                directoryName: "owner--pack",
                cacheRoot: URL(fileURLWithPath: "/models")
            ),
            [
                "remove", "owner--pack", "--yes", "--missing-ok", "--json",
                "--cache-dir", "/models",
            ]
        )
    }

    func testCachedEntryNameOnlyAcceptsDirectManagedChildren() {
        let downloader = ModelDownloader(
            processEnvironment: ["HOME": "/Users/test"],
            modelCacheRoot: URL(fileURLWithPath: "/models", isDirectory: true)
        )

        XCTAssertEqual(
            downloader.cachedEntryName(forInstalledPath: "/models/owner--pack"),
            "owner--pack"
        )
        XCTAssertEqual(
            downloader.cachedModelReference(forInstalledPath: "/models/owner--pack"),
            "owner/pack"
        )
        XCTAssertNil(
            downloader.cachedEntryName(forInstalledPath: "/models/nested/owner--pack")
        )
        XCTAssertNil(
            downloader.cachedEntryName(forInstalledPath: "/Users/test/models/owner--pack")
        )
        XCTAssertNil(downloader.cachedEntryName(forInstalledPath: "/models"))
        XCTAssertNil(downloader.cachedEntryName(forInstalledPath: "/models/.cache"))
        XCTAssertNil(downloader.cachedEntryName(forInstalledPath: "/models/owner--pack/../.."))
    }

    func testCachedEntryNameRefusesSymlinkedUserManagedModel() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-remove-link-\(UUID().uuidString)", isDirectory: true)
        let cache = root.appendingPathComponent("cache", isDirectory: true)
        let external = root.appendingPathComponent("external", isDirectory: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let link = cache.appendingPathComponent("owner--pack", isDirectory: true)
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: external)
        let downloader = ModelDownloader(modelCacheRoot: cache)

        XCTAssertNil(downloader.cachedEntryName(forInstalledPath: link.path))
        XCTAssertNil(downloader.cachedModelReference(forInstalledPath: link.path))
    }

    func testRemovalRunsCLIAndDecodesFreedBytes() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-remove-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let executable = try makeFakeCLI(
            in: root,
            printing: #"{"repo_id":"owner/pack","path":"\#(cacheRoot.path)/owner--pack","removed":true,"size_bytes_removed":1234}"#
        )
        let argumentsLog = root.appendingPathComponent("arguments.log")
        let stdinLog = root.appendingPathComponent("stdin.log")
        let downloader = ModelDownloader(
            processEnvironment: [
                "HOME": root.path,
                "MTPLX_ARGUMENTS_LOG": argumentsLog.path,
                "MTPLX_STDIN_LOG": stdinLog.path,
            ],
            modelCacheRoot: cacheRoot,
            executableOverride: executable
        )

        let result = try await downloader.removeCachedModel(directoryName: "owner--pack")

        XCTAssertTrue(result.removed)
        XCTAssertEqual(result.sizeBytesRemoved, 1_234)
        XCTAssertEqual(
            try String(contentsOf: argumentsLog, encoding: .utf8),
            "remove owner--pack --yes --missing-ok --json --cache-dir \(cacheRoot.standardizedFileURL.path)"
        )
        // stdin is /dev/null: the CLI can never block on a prompt.
        XCTAssertEqual(try String(contentsOf: stdinLog, encoding: .utf8), "eof")
    }

    func testRemovalRejectsACLIResultForAnotherEntry() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-remove-mismatch-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let executable = try makeFakeCLI(
            in: root,
            printing: #"{"repo_id":"other/pack","path":"\#(cacheRoot.path)/other--pack","removed":true,"size_bytes_removed":1}"#
        )
        let downloader = ModelDownloader(
            processEnvironment: ["HOME": root.path],
            modelCacheRoot: cacheRoot,
            executableOverride: executable
        )

        do {
            _ = try await downloader.removeCachedModel(directoryName: "owner--pack")
            XCTFail("expected the mismatched result to be refused")
        } catch let error as NSError {
            XCTAssertEqual(error.domain, "ModelDownloader")
            XCTAssertEqual(error.code, 3)
        }
    }

    func testRemovalSurfacesTheCLIRefusal() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-remove-refusal-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let executable = root.appendingPathComponent("mtplx")
        let script = """
        #!/bin/sh
        echo 'mtplx remove: refusing to remove /cache: model ref does not name a model directory' >&2
        exit 1
        """
        try Data(script.utf8).write(to: executable)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: executable.path)
        let downloader = ModelDownloader(
            processEnvironment: ["HOME": root.path],
            modelCacheRoot: root.appendingPathComponent("cache", isDirectory: true),
            executableOverride: executable
        )

        do {
            _ = try await downloader.removeCachedModel(directoryName: "owner--pack")
            XCTFail("expected the CLI refusal to surface")
        } catch let error as NSError {
            XCTAssertEqual(error.code, 1)
            XCTAssertTrue(error.localizedDescription.contains("refusing to remove"), error.localizedDescription)
        }
    }

    @MainActor
    func testBackendRefusesToRemoveSelectedModel() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-selected-remove-\(UUID().uuidString)", isDirectory: true)
        let installed = root.appendingPathComponent("owner--pack", isDirectory: true)
        var configuration = MTPLXAppConfiguration()
        configuration.model = installed.path
        let store = MTPLXBackendStore(
            configuration: configuration,
            settingsStore: isolatedSettingsStore(in: root),
            modelDownloader: ModelDownloader(modelCacheRoot: root)
        )

        do {
            _ = try await store.removeCachedModel(
                repoID: "owner/pack",
                installedPath: installed.path
            )
            XCTFail("expected selected-model refusal")
        } catch let error as CachedModelRemovalError {
            XCTAssertEqual(error, .selectedModel)
        }
    }

    @MainActor
    func testBackendRefusesAnEntryOutsideTheManagedCache() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mtplx-outside-remove-\(UUID().uuidString)", isDirectory: true)
        let store = MTPLXBackendStore(
            configuration: MTPLXAppConfiguration(),
            settingsStore: isolatedSettingsStore(in: root),
            modelDownloader: ModelDownloader(modelCacheRoot: root.appendingPathComponent("cache", isDirectory: true))
        )

        do {
            _ = try await store.removeCachedModel(
                repoID: "owner/pack",
                installedPath: root.appendingPathComponent("elsewhere/owner--pack").path
            )
            XCTFail("expected outside-cache refusal")
        } catch let error as CachedModelRemovalError {
            XCTAssertEqual(error, .outsideManagedCache)
        }
    }

    /// The daemon keeps its weights mapped while it runs, so the entry it
    /// reports on /health is protected even when the selection has already
    /// moved on (a switch to a model that still has to download).
    func testRunningDaemonProtectsTheEntryItServes() {
        let installed = "/Users/test/.mtplx/models/owner--pack"
        for state in [DaemonState.running, .starting, .warming, .stopping, .degraded("lost")] {
            XCTAssertTrue(
                MTPLXBackendStore.runningDaemonReads(
                    installedPath: installed,
                    repoID: "owner/pack",
                    daemonState: state,
                    healthModel: "owner/pack",
                    healthModelPath: "/Users/test/.mtplx/models/owner--pack/"
                ),
                "\(state)"
            )
        }
        XCTAssertTrue(
            MTPLXBackendStore.runningDaemonReads(
                installedPath: installed,
                repoID: "owner/pack",
                daemonState: .running,
                healthModel: "OWNER/PACK",
                healthModelPath: "/somewhere/else"
            ),
            "the served model id matches the entry's reference"
        )
        XCTAssertFalse(
            MTPLXBackendStore.runningDaemonReads(
                installedPath: installed,
                repoID: "owner/pack",
                daemonState: .running,
                healthModel: "other/pack",
                healthModelPath: "/Users/test/.mtplx/models/other--pack"
            ),
            "a daemon on another model does not block"
        )
        for state in [DaemonState.stopped, .crashed(nil)] {
            XCTAssertFalse(
                MTPLXBackendStore.runningDaemonReads(
                    installedPath: installed,
                    repoID: "owner/pack",
                    daemonState: state,
                    healthModel: "owner/pack",
                    healthModelPath: installed
                ),
                "\(state)"
            )
        }
        XCTAssertFalse(
            MTPLXBackendStore.runningDaemonReads(
                installedPath: installed,
                repoID: "owner/pack",
                daemonState: .running,
                healthModel: nil,
                healthModelPath: nil
            ),
            "no health yet: the selected-model guard covers startup"
        )
    }

    // MARK: - Helpers

    private func isolatedSettingsStore(in root: URL) -> MTPLXSettingsStore {
        MTPLXSettingsStore(settingsURL: root.appendingPathComponent("settings.json"))
    }

    /// A stand-in `mtplx` that records its argument vector and whether stdin
    /// had anything to read, then prints the given JSON.
    private func makeFakeCLI(in root: URL, printing json: String) throws -> URL {
        let executable = root.appendingPathComponent("mtplx")
        let script = """
        #!/bin/sh
        if [ -n "${MTPLX_ARGUMENTS_LOG:-}" ]; then printf '%s' "$*" > "$MTPLX_ARGUMENTS_LOG"; fi
        if [ -n "${MTPLX_STDIN_LOG:-}" ]; then
            if IFS= read -r _line; then printf 'data' > "$MTPLX_STDIN_LOG"; else printf 'eof' > "$MTPLX_STDIN_LOG"; fi
        fi
        printf '%s\\n' '\(json)'
        """
        try Data(script.utf8).write(to: executable)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755],
            ofItemAtPath: executable.path
        )
        return executable
    }
}
