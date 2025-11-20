const controls = window;
const mpFaceMesh = window;
const THREE = window.THREE;
const config = {
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    }
};
// Our input frames will come from here.
const videoElement = document.getElementsByClassName("input_video")[0];
const canvasElement = document.getElementsByClassName("output_canvas")[0];
const controlsElement = document.getElementsByClassName("control-panel")[0];
/**
 * Solution options.
 */
const solutionOptions = {
    selfieMode: true,
    enableFaceGeometry: true,
    maxNumFaces: 1,
    refineLandmarks: false,
    minDetectionConfidence: 0.8,
    minTrackingConfidence: 0.8,
    modelScale: 8,
    modelPositionY: 0
};
// We'll add this to our control panel later, but we'll save it here so we can
// call tick() each time the graph runs.
const fpsControl = new controls.FPS();
// Optimization: Turn off animated spinner after its hiding animation is done.
const spinner = document.querySelector(".loading");
spinner.ontransitionend = () => {
    spinner.style.display = "none";
};
const smoothingFactor = (te, cutoff) => {
    const r = 2 * Math.PI * cutoff * te;
    return r / (r + 1);
};
const exponentialSmoothing = (a, x, xPrev) => {
    return a * x + (1 - a) * xPrev;
};
class OneEuroFilter {
    constructor({ minCutOff, beta }) {
        this.minCutOff = minCutOff;
        this.beta = beta;
        this.dCutOff = 0.001; // period in milliseconds, so default to 0.001 = 1Hz
        this.xPrev = null;
        this.dxPrev = null;
        this.tPrev = null;
        this.initialized = false;
    }
    reset() {
        this.initialized = false;
    }
    filter(t, x) {
        if (!this.initialized) {
            this.initialized = true;
            this.xPrev = x;
            this.dxPrev = x.map(() => 0);
            this.tPrev = t;
            return x;
        }
        const { xPrev, tPrev, dxPrev } = this;
        //console.log("filter", x, xPrev, x.map((xx, i) => x[i] - xPrev[i]));
        const te = t - tPrev;
        const ad = smoothingFactor(te, this.dCutOff);
        const dx = [];
        const dxHat = [];
        const xHat = [];
        for (let i = 0; i < x.length; i++) {
            // The filtered derivative of the signal.
            dx[i] = (x[i] - xPrev[i]) / te;
            dxHat[i] = exponentialSmoothing(ad, dx[i], dxPrev[i]);
            // The filtered signal
            const cutOff = this.minCutOff + this.beta * Math.abs(dxHat[i]);
            const a = smoothingFactor(te, cutOff);
            xHat[i] = exponentialSmoothing(a, x[i], xPrev[i]);
        }
        // update prev
        this.xPrev = xHat;
        this.dxPrev = dxHat;
        this.tPrev = t;
        return xHat;
    }
}
class EffectRenderer {
    constructor() {
        this.VIDEO_DEPTH = 500;
        this.FOV_DEGREES = 63;
        this.NEAR = 1;
        this.FAR = 10000;
        this.loadedModel = null;
        this.matrixX = [];
        this.scene = new THREE.Scene();
        this.filters = new OneEuroFilter({ minCutOff: 0.001, beta: 1 });
        this.renderer = new THREE.WebGLRenderer({
            alpha: true,
            antialias: true,
            stencilBuffer: true,
            canvas: canvasElement,
            context: canvasElement.getContext('webgl2')
        });
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.shadowMap.enabled = false;
        this.renderer.gammaFactor = 0;
        const targetObject = new THREE.Object3D();
        targetObject.position.set(0, 0, -1);
        this.scene.add(targetObject);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.castShadow = true;
        directionalLight.position.set(0, 0.25, 0);
        directionalLight.target = targetObject;
        this.scene.add(directionalLight);
        const bounceLight = new THREE.HemisphereLight(0xffffff, 0xffffff, 0.5);
        this.scene.add(bounceLight);
        this.faceGroup = new THREE.Group();
        this.faceGroup.matrixAutoUpdate = false;
        this.scene.add(this.faceGroup);
        const loader = new THREE.GLTFLoader();
        loader.load("result.gltf", (gltf) => {
            console.log("Loaded result.gltf:", gltf);
            const scene = gltf.scene;
            scene.traverse((node) => {
                if (node.isMesh) {
                    node.renderOrder = 3;
                }
            });
            // Store reference to the loaded model
            this.loadedModel = gltf.scene;
            // Set initial position and scale
            this.loadedModel.position.set(0, solutionOptions.modelPositionY, 0);
            this.loadedModel.scale.set(solutionOptions.modelScale, solutionOptions.modelScale, solutionOptions.modelScale);
            this.faceGroup.add(this.loadedModel);
        }, undefined, (error) => {
            console.error("Error loading result.gltf:", error);
        });
    }
    updateModelTransform() {
        if (this.loadedModel) {
            this.loadedModel.scale.set(solutionOptions.modelScale, solutionOptions.modelScale, solutionOptions.modelScale);
            this.loadedModel.position.y = solutionOptions.modelPositionY;
        }
    }
    async render(results) {
        this.onCanvasDimsUpdate();
        this.updateModelTransform();
        console.log();
        const image = await createImageBitmap(results.image);
        const imagePlane = this.createGpuBufferPlane(results.image);
        this.scene.add(imagePlane);
        if (results.multiFaceGeometry.length > 0) {
            const faceGeometry = results.multiFaceGeometry[0];
            const poseTransformMatrixData = faceGeometry.getPoseTransformMatrix();
            this.faceGroup.matrix.fromArray(this.filters.filter(Date.now(), poseTransformMatrixData.getPackedDataList()));
            this.faceGroup.visible = true;
        }
        else {
            this.filters.reset();
            this.faceGroup.visible = false;
        }
        this.renderer.render(this.scene, this.camera);
        this.scene.remove(imagePlane);
    }
    createGpuBufferPlane(gpuBuffer) {
        const depth = this.VIDEO_DEPTH;
        const fov = this.camera.fov;
        const width = canvasElement.width;
        const height = canvasElement.height;
        const aspect = width / height;
        const viewportHeightAtDepth = 2 * depth * Math.tan(THREE.MathUtils.degToRad(0.5 * fov));
        const viewportWidthAtDepth = viewportHeightAtDepth * aspect;
        console.log(viewportHeightAtDepth, viewportWidthAtDepth);
        const texture = new THREE.CanvasTexture(gpuBuffer);
        texture.minFilter = THREE.LinearFilter;
        texture.encoding = THREE.sRGBEncoding;
        const plane = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: texture }));
        plane.scale.set(viewportWidthAtDepth, viewportHeightAtDepth, 1);
        plane.position.set(0, 0, -depth);
        return plane;
    }
    onCanvasDimsUpdate() {
        this.camera = new THREE.PerspectiveCamera(this.FOV_DEGREES, canvasElement.width / canvasElement.height, this.NEAR, this.FAR);
        this.renderer.setSize(canvasElement.width, canvasElement.height);
    }
}
const effectRenderer = new EffectRenderer();
function onResults(results) {
    // Hide the spinner.
    document.body.classList.add("loaded");
    // Render the effect.
    effectRenderer.render(results);
    // Update the frame rate.
    fpsControl.tick();
}
const faceMesh = new mpFaceMesh.FaceMesh(config);
faceMesh.setOptions(solutionOptions);
faceMesh.onResults(onResults);
// Present a control panel through which the user can manipulate the solution
// options.
new controls.ControlPanel(controlsElement, solutionOptions)
    .add([
    new controls.StaticText({ title: "MediaPipe + Three.JS" }),
    fpsControl,
    new controls.Toggle({ title: "Selfie Mode", field: "selfieMode" }),
    new controls.Toggle({
        title: "Face Transform",
        field: "enableFaceGeometry"
    }),
    new controls.SourcePicker({
        onFrame: async (input, size) => {
            const aspect = size.height / size.width;
            let width, height;
            if (window.innerWidth > window.innerHeight) {
                height = window.innerHeight;
                width = height / aspect;
            }
            else {
                width = window.innerWidth;
                height = width * aspect;
            }
            canvasElement.width = width;
            canvasElement.height = height;
            await faceMesh.send({ image: input });
        }
    }),
    new controls.Slider({
        title: "Max Number of Faces",
        field: "maxNumFaces",
        range: [1, 4],
        step: 1
    }),
    new controls.Toggle({
        title: "Refine Landmarks",
        field: "refineLandmarks"
    }),
    new controls.Slider({
        title: "Min Detection Confidence",
        field: "minDetectionConfidence",
        range: [0, 1],
        step: 0.01
    }),
    new controls.Slider({
        title: "Min Tracking Confidence",
        field: "minTrackingConfidence",
        range: [0, 1],
        step: 0.01
    }),
    new controls.Slider({
        title: "Model Scale",
        field: "modelScale",
        range: [0.1, 20],
        step: 0.1
    }),
    new controls.Slider({
        title: "Model Position Y",
        field: "modelPositionY",
        range: [-10, 10],
        step: 0.1
    })
])
    .on((x) => {
    const options = x;
    videoElement.classList.toggle("selfie", options.selfieMode);
    faceMesh.setOptions(options);
});
export {};
