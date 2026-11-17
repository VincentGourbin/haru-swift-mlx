// HaruStudioApp.swift - Haru Studio: talk with Haru-4B on this Mac, model in-process (MLX)
// Copyright 2026 Vincent Gourbin

import AppKit
import SwiftUI

@main
struct HaruStudioApp: App {
    @State private var model = StudioModel()

    init() {
        // Also usable unbundled (`swift run`-style launch): become a regular foreground app.
        NSApplication.shared.setActivationPolicy(.regular)
    }

    var body: some Scene {
        WindowGroup("Haru Studio") {
            ContentView(model: model)
                .frame(minWidth: 860, minHeight: 620)
                .onAppear { NSApplication.shared.activate() }
        }
    }
}
