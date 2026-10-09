import QuartzCore
import SwiftUI
import UIKit

final class MetalView: UIView {
    override class var layerClass: AnyClass { CAMetalLayer.self }

    var metalLayer: CAMetalLayer { layer as! CAMetalLayer }

    override func layoutSubviews() {
        super.layoutSubviews()
        let scale = window?.screen.nativeScale ?? UIScreen.main.nativeScale
        metalLayer.contentsScale = scale
        metalLayer.drawableSize = CGSize(width: bounds.width * scale, height: bounds.height * scale)
    }
}

/// Hosts the Metal layer and maps gestures to controls. The Cardboard v2 button is a single tap.
///   tap: reset tracking (map + position) · 2-finger tap: HUD on/off · 3-finger tap: reprojection
///   long press: calibration (the only place to change tracking mode / camera, so it can't change by accident)
final class VRViewController: UIViewController {
    private let model: AppModel

    init(model: AppModel) {
        self.model = model
        super.init(nibName: nil, bundle: nil)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not supported") }

    override func loadView() {
        view = MetalView()
        view.backgroundColor = .black
        view.isMultipleTouchEnabled = true
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        // No double-tap recognizer: a single tap must fire immediately, not after a double-tap timeout.
        _ = tap(#selector(onTap), taps: 1, touches: 1)
        _ = tap(#selector(onTwoFingerTap), taps: 1, touches: 2)
        _ = tap(#selector(onThreeFingerTap), taps: 1, touches: 3)

        let longPress = UILongPressGestureRecognizer(target: self, action: #selector(onLongPress))
        longPress.minimumPressDuration = 0.8
        view.addGestureRecognizer(longPress)

        let center = NotificationCenter.default
        center.addObserver(self, selector: #selector(onActive), name: UIApplication.didBecomeActiveNotification, object: nil)
        center.addObserver(self, selector: #selector(onInactive), name: UIApplication.willResignActiveNotification, object: nil)
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        view.setNeedsLayout()
        view.layoutIfNeeded()
        if let view = view as? MetalView { model.engine?.attach(to: view.metalLayer) }
        model.setVRScreen(view.window?.windowScene?.screen)
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        model.setVRScreen(nil)
    }

    @objc private func onActive() {
        if view.window != nil { model.setVRScreen(view.window?.windowScene?.screen) }
    }

    @objc private func onInactive() { model.setVRScreen(nil) }

    override var prefersHomeIndicatorAutoHidden: Bool { true }
    override var prefersStatusBarHidden: Bool { true }
    override var supportedInterfaceOrientations: UIInterfaceOrientationMask { .landscapeRight }

    private func tap(_ action: Selector, taps: Int, touches: Int) -> UITapGestureRecognizer {
        let recognizer = UITapGestureRecognizer(target: self, action: action)
        recognizer.numberOfTapsRequired = taps
        recognizer.numberOfTouchesRequired = touches
        view.addGestureRecognizer(recognizer)
        return recognizer
    }

    @objc private func onTap() { model.resetTracking() }
    @objc private func onTwoFingerTap() { model.toggleHUD() }
    @objc private func onThreeFingerTap() { model.toggleReprojection() }

    @objc private func onLongPress(_ recognizer: UILongPressGestureRecognizer) {
        if recognizer.state == .began { model.openCalibration() }
    }
}

struct VRContainer: UIViewControllerRepresentable {
    let model: AppModel

    func makeUIViewController(context: Context) -> VRViewController { VRViewController(model: model) }
    func updateUIViewController(_ controller: VRViewController, context: Context) {}
}
