// @ts-ignore - CDN import not recognized by TypeScript
// Importa a biblioteca de filtro Kalman para suavização de dados
import * as kalmanFilter from "https://cdn.skypack.dev/kalman-filter@1.10.1";

// Declara os tipos globais para as bibliotecas carregadas via CDN no HTML
declare global {
  interface Window {
    THREE: any; // Three.js para renderização 3D
    FPS: any;
    ControlPanel: any;
    StaticText: any;
    Toggle: any;
    Slider: any;
    SourcePicker: any;
    InputImage: any;
    Rectangle: any;
    FaceMesh: any;
    Results: any;
    Options: any;
    GpuBuffer: any;
  }
}

const controls: any = window;
const mpFaceMesh: any = window;
const THREE: any = window.THREE;

// Configuração para localizar os arquivos WASM do MediaPipe
const config = {
  locateFile: (file) => {
    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
  }
};

// Frames de input serão pegos daqui
const videoElement = document.getElementsByClassName(
  "input_video"
)[0] as HTMLVideoElement;
const canvasElement = document.getElementsByClassName(
  "output_canvas"
)[0] as HTMLCanvasElement;
const controlsElement = document.getElementsByClassName(
  "control-panel"
)[0] as HTMLDivElement;

/**
 * Configurações do MediaPipe Face Mesh e do modelo 3D que vai aparecer na tela
 */
const solutionOptions = {
  selfieMode: true,
  enableFaceGeometry: true,
  maxNumFaces: 1,
  refineLandmarks: false,
  minDetectionConfidence: 0.8,
  minTrackingConfidence: 0.8,
  modelScale: 12.4,
  modelPositionY: 4.9
};


const fpsControl = new controls.FPS();
const spinner = document.querySelector(".loading")! as HTMLDivElement;
spinner.ontransitionend = () => {
  spinner.style.display = "none";
};

// Calcula o fator de suavização baseado no tempo e frequência de corte
const smoothingFactor = (te, cutoff) => {
  const r = 2 * Math.PI * cutoff * te;
  return r / (r+1);
}

const exponentialSmoothing = (a, x, xPrev) => {
  return a * x + (1 - a) * xPrev;
}

/**
 * One Euro Filter - Reduz tremores (jitter) nos dados de rastreamento
 * Mantém responsividade em movimentos rápidos e estabilidade em movimentos lentos
 */
class OneEuroFilter {
  private minCutOff: number;
  private beta: number;
  private dCutOff: number;
  private xPrev: number[] | null;
  private dxPrev: number[] | null;
  private tPrev: number | null;
  private initialized: boolean;

  constructor({minCutOff, beta}: {minCutOff: number, beta: number}) {
    this.minCutOff = minCutOff;
    this.beta = beta;
    this.dCutOff = 0.001;

    this.xPrev = null;
    this.dxPrev = null;
    this.tPrev = null;
    this.initialized = false;
  }

  reset() {
    this.initialized = false;
  }

  filter(t, x) {

    // Se não estiver inicializado, inicializa com os valores atuais
    if (!this.initialized) {
      this.initialized = true;
      this.xPrev = x;
      this.dxPrev = x.map(() => 0);
      this.tPrev = t;
      return x;
    }

    // Guarda os dados anteriores para calcular a diferença entre o valor atual e o anterior
    const {xPrev, tPrev, dxPrev} = this;

    //console.log("filter", x, xPrev, x.map((xx, i) => x[i] - xPrev[i]));

    const te = t - tPrev;

    const ad = smoothingFactor(te, this.dCutOff);

    const dx = [];
    const dxHat = [];
    const xHat = [];
    for (let i = 0; i < x.length; i++) {
      // Calcula a diferença entre o valor atual e o anterior
      dx[i] = (x[i] - xPrev[i]) / te;
      dxHat[i] = exponentialSmoothing(ad, dx[i], dxPrev[i]);

      const cutOff = this.minCutOff + this.beta * Math.abs(dxHat[i]);
      const a = smoothingFactor(te, cutOff);
      xHat[i] = exponentialSmoothing(a, x[i], xPrev[i]);
    }

    this.xPrev = xHat; 
    this.dxPrev = dxHat;
    this.tPrev = t;

    return xHat;
  }
}

class EffectRenderer {
  // Configurações da câmera virtual 3D
  private readonly VIDEO_DEPTH = 500;
  private readonly FOV_DEGREES = 63;
  private readonly NEAR = 1;
  private readonly FAR = 10000;

  private readonly scene: any;           // Cena Three.js
  private readonly renderer: any;        // Renderizador WebGL
  private readonly faceGroup: any;       // Grupo que contém o modelo 3D
  private loadedModel: any = null;       // Referência ao modelo carregado

  private camera: any;
  private matrixX = [];
  private filters;

  constructor() {
    // Inicializa a cena Three.js
    this.scene = new THREE.Scene();
    
    // Filtro para suavizar os dados da matriz de transformação
    this.filters = new OneEuroFilter({minCutOff: 0.001, beta: 1});

    // Configura o renderizador WebGL com transparência
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

    // Configuração de iluminação
    const targetObject = new THREE.Object3D();
    targetObject.position.set(0, 0, -1);
    this.scene.add(targetObject);

    // Luz direcional
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.castShadow = true;
    directionalLight.position.set(0, 0.25, 0);
    directionalLight.target = targetObject;
    this.scene.add(directionalLight);
    
    // Luz hemisférica (iluminação ambiente)
    const bounceLight = new THREE.HemisphereLight(0xffffff, 0xffffff, 0.5);
    this.scene.add(bounceLight);

    // Grupo que seguirá a transformação do rosto
    this.faceGroup = new THREE.Group();
    this.faceGroup.matrixAutoUpdate = false;
    this.scene.add(this.faceGroup);

    // Carrega o modelo 3D GLTF
    const loader = new THREE.GLTFLoader();
    loader.load("result.gltf", (gltf) => {
      console.log("Loaded result.gltf:", gltf);
      
      const scene = gltf.scene;
      scene.traverse((node) => {
        if (node.isMesh) {
          node.renderOrder = 3;
        }
      });
      
      // Armazena referência ao modelo
      this.loadedModel = gltf.scene;
      
      // Aplica posição e escala inicial
      this.loadedModel.position.set(0, solutionOptions.modelPositionY, 0);
      this.loadedModel.scale.set(
        solutionOptions.modelScale, 
        solutionOptions.modelScale, 
        solutionOptions.modelScale
      );
      
      this.faceGroup.add(this.loadedModel);
    }, undefined, (error) => {
      console.error("Error loading result.gltf:", error);
    });
  }

  /**
   * Atualiza a transformação do modelo baseado nas opções
   * (chamado a cada frame para permitir ajustes em tempo real)
   */
  updateModelTransform() {
    if (this.loadedModel) {
      this.loadedModel.scale.set(
        solutionOptions.modelScale,
        solutionOptions.modelScale,
        solutionOptions.modelScale
      );
      this.loadedModel.position.y = solutionOptions.modelPositionY;
    }
  }

  /**
   * Renderiza um frame da cena 3D
   * @param results Resultados do MediaPipe com dados do rosto
   */
  async render(results: any) {
    this.onCanvasDimsUpdate();
    this.updateModelTransform();
    
    console.log()
    
    // Cria um plano com a imagem da webcam como textura
    const image = await createImageBitmap(results.image);
    const imagePlane = this.createGpuBufferPlane(results.image);
    this.scene.add(imagePlane);

    // Se detectou um rosto, aplica a transformação 3D
    if (results.multiFaceGeometry.length > 0) {
      const faceGeometry = results.multiFaceGeometry[0];

      // Obtém a matriz 4x4 de transformação do rosto (posição, rotação, escala)
      const poseTransformMatrixData = faceGeometry.getPoseTransformMatrix();
      
      // Aplica o filtro de suavização e atualiza a matriz do grupo
      this.faceGroup.matrix.fromArray(
        this.filters.filter(Date.now(), poseTransformMatrixData.getPackedDataList())
      );
      this.faceGroup.visible = true;
    } else {
      // Nenhum rosto detectado: reseta o filtro e esconde o modelo
      this.filters.reset();
      this.faceGroup.visible = false;
    }

    // Renderiza a cena
    this.renderer.render(this.scene, this.camera);

    // Remove o plano da imagem (será recriado no próximo frame)
    this.scene.remove(imagePlane);
  }

  /**
   * Cria um plano 3D com a imagem da webcam como textura de fundo
   */
  private createGpuBufferPlane(gpuBuffer: any): any {
    const depth = this.VIDEO_DEPTH;
    const fov = this.camera.fov;

    const width = canvasElement.width;
    const height = canvasElement.height;
    const aspect = width / height;

    // Calcula o tamanho do plano para preencher o campo de visão
    const viewportHeightAtDepth =
      2 * depth * Math.tan(THREE.MathUtils.degToRad(0.5 * fov));
    const viewportWidthAtDepth = viewportHeightAtDepth * aspect;
    
    console.log(viewportHeightAtDepth, viewportWidthAtDepth);

    // Cria textura a partir da imagem da webcam
    const texture = new THREE.CanvasTexture(gpuBuffer as HTMLCanvasElement);
    texture.minFilter = THREE.LinearFilter;
    texture.encoding = THREE.sRGBEncoding;

    // Cria um plano com a textura
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: texture })
    );

    plane.scale.set(viewportWidthAtDepth, viewportHeightAtDepth, 1);
    plane.position.set(0, 0, -depth);

    return plane;
  }

  /**
   * Atualiza as dimensões da câmera quando o canvas muda de tamanho
   */
  private onCanvasDimsUpdate() {
    this.camera = new THREE.PerspectiveCamera(
      this.FOV_DEGREES,
      canvasElement.width / canvasElement.height,
      this.NEAR,
      this.FAR
    );

    this.renderer.setSize(canvasElement.width, canvasElement.height);
  }
}

// Cria a instância do renderizador
const effectRenderer = new EffectRenderer();

function onResults(results: any): void {
  document.body.classList.add("loaded");

  effectRenderer.render(results);

  fpsControl.tick();
}

const faceMesh = new mpFaceMesh.FaceMesh(config);
faceMesh.setOptions(solutionOptions);
faceMesh.onResults(onResults);

// Mostra painel de controle para ajustar as configurações
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
      onFrame: async (input: any, size: any) => {
        const aspect = size.height / size.width;
        let width: number, height: number;
        if (window.innerWidth > window.innerHeight) {
          height = window.innerHeight;
          width = height / aspect;
        } else {
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
    const options = x as any;
    videoElement.classList.toggle("selfie", options.selfieMode);
    faceMesh.setOptions(options);
  });
