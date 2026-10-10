# SplatForge

App desktop (Tauri 2 + React) que transforma **vídeos orbitais** — inclusive
gerados por IA — em **Gaussian Splatting**, 100% local, no macOS (Apple
Silicon) e no Windows (com aceleração CUDA no COLMAP quando há GPU NVIDIA).

```
vídeo ──► Frames ─────────► Recorte ─────────► Poses ──────────────► Treino ─────────► splat.ply
          ffmpeg            (opcional)          COLMAP 4              Brush
          decodificação     Vision (macOS),     SIFT reforçado,       wgpu (Metal,
          por hardware +    BiRefNet+DirectML   matching circular,    Vulkan/DX12),
          nitidez (Rust)    (Windows)           mapeador global       prévias ao vivo
```

## Instalação

### macOS

```bash
brew install ffmpeg colmap
pnpm install
pnpm setup:tools        # baixa o Brush (sidecar do Tauri) e confere ffmpeg/COLMAP
```

### Windows (usuário final)

1. Baixe e execute o `SplatForge_<versão>_x64-setup.exe` (gerado pelo
   GitHub Actions, ver abaixo). Ele instala só para o usuário atual, sem
   pedir administrador, e baixa o WebView2 se faltar.
2. Na primeira execução o app mostra **"Instalar ferramentas"**: baixa o
   COLMAP (com CUDA se houver GPU NVIDIA) e o ffmpeg, confere o SHA-256 e
   instala em `%LOCALAPPDATA%\SplatForge\tools`. O Brush já vem no
   instalador.

Sem assinatura de código, o Windows SmartScreen mostra "O Windows protegeu
o computador": clique em **Mais informações → Executar assim mesmo**. Para
eliminar o aviso é preciso assinar o instalador (certificado de code signing
ou Azure Trusted Signing).

### Windows (desenvolvimento)

Pré-requisitos: Node 20+, pnpm, Rust com o toolchain MSVC (Visual Studio
Build Tools, carga "Desenvolvimento para desktop com C++") e WebView2 (já vem
no Windows 11). Para CUDA, um driver NVIDIA recente.

```powershell
pnpm install
pnpm setup:tools        # Brush + COLMAP 4.2.1 + ffmpeg, com SHA-256 conferido
```

O `setup:tools` escolhe o COLMAP com CUDA se encontrar `nvidia-smi` (use
`--cpu` para forçar a versão sem CUDA) e instala COLMAP e ffmpeg em
`%LOCALAPPDATA%\SplatForge\tools`, onde o app procura. Também são
encontrados executáveis instalados por winget, scoop, Chocolatey ou no PATH.

Com CUDA, extração SIFT, matching e bundle adjustment do COLMAP rodam na GPU.
O SIFT reforçado (forma afim + agrupamento de escalas) só existe na CPU; o
COLMAP faz esse fallback sozinho e o matching continua na GPU.

## Uso

```bash
pnpm tauri dev          # desenvolvimento
pnpm tauri build        # macOS: .app e .dmg · Windows: instalador NSIS
```

### Gerar os instaladores no GitHub

O workflow `.github/workflows/release.yml` compila em máquinas Windows e
macOS do GitHub e publica os instaladores numa release de rascunho:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Ou dispare manualmente em **Actions → release → Run workflow**.

1. Arraste um vídeo para a janela.
2. Se o objeto "derrete" no fim da órbita, marque um trecho estável com
   **Início aqui / Fim aqui**.
3. Escolha um preset e clique em **Gerar splat**.

### Recorte do sujeito

Com **"Isolar o sujeito"** ligado, cada frame ganha uma máscara do sujeito e o
Brush treina com o fundo transparente: o splat sai só com a pessoa ou objeto,
sem paredes, pessoas ao fundo ou os riscos em volta.

| Método | Sistema | Como | Medido |
|---|---|---|---|
| IA (Vision) | macOS 14+ | Recorte nativo do sistema, na GPU, via o sidecar `splatforge_mask` (compilado pelo `build.rs`) | ~40 ms/frame num M4 |
| Pessoas | Windows | MODNet (Apache 2.0) com ONNX Runtime + DirectML | 144 ms/frame na CPU de um M4; GPU não medida |
| Objetos | Windows | BiRefNet-lite (MIT) com ONNX Runtime + DirectML | ~4 s/frame numa RTX 4070 Ti Super |
| RMBG | Windows, macOS (Apple Silicon) | RMBG-1.4 (BRIA AI, **não comercial**) com ONNX Runtime + DirectML / CoreML na GPU | ~0,4 s/frame num M4 (+~5 s para compilar no CoreML) |
| Cor sólida | todos | Chroma key em Rust: distância RGB à cor de fundo (detectada pela mediana das bordas ou escolhida; áreas mais iluminadas da tela também contam), tolerância, suavização e preenchimento a partir das bordas | 120 frames em 0,09 s |

O BiRefNet é lento em qualquer GPU porque as convoluções deformáveis viram
~16 mil operações minúsculas no ONNX; o MODNet é uma rede convolucional
simples e, nos frames testados (pessoas de frente e de costas), recortou tão
bem quanto ele. O RMBG-1.4 tem bordas mais limpas que o Vision (sem o halo
claro em volta do cabelo), mas a licença é não comercial: por isso ele só é
baixado (176 MB, por **"Instalar ferramentas"**) quando escolhido. No macOS
roda no CoreML só com CPU e GPU: compilar para o Neural Engine pode travar o
`ANECompilerService` do sistema.

No Windows os modelos (MODNet 26 MB, BiRefNet 224 MB) são baixados por
**"Instalar ferramentas"** e o `DirectML.dll` 1.15.4 vai dentro do instalador.

**Despill** (qualquer método): com fundo verde ou azul, a luz rebatida pela
tela tinge o sujeito inteiro, não só a borda. A cor do fundo é a mediana dos
pixels que a máscara marca como fundo; em todo o sujeito o canal dessa cor é
limitado ao maior dos outros dois e a luminância tirada volta como cinza (o
sujeito não escurece; amarelos e cianos ficam intactos). Fundos pouco
saturados ou de outra cor são ignorados. Numa tela verde com sujeito de cor
parecida, **RMBG + despill** dá o melhor resultado: a IA faz o recorte e o
despill tira o verde.

Opcionalmente as máscaras também vão para o COLMAP ("Usar o recorte também
nas poses"): ajuda quando o fundo muda entre frames, mas tira pontos de apoio
de sujeitos com pouca textura, por isso vem desligado.

Cada vídeo ganha uma pasta de projeto (`<video>_splat/` por padrão). Para
mudar só os parâmetros de treino, escolha **Executar a partir de → Treino**.
Assim frames e poses são reaproveitados.

```
<video>_splat/
├── project.json        estado (frames, métricas, resultado)
├── dataset/            entrada do Brush: images/ + sparse/0/
├── masks/              recortes (cinza, 255 = sujeito)
├── dataset_alpha/      com recorte: PNGs RGBA + images.bin com nomes .png
├── colmap/             database.db, pairs.txt, modelos brutos
├── training/           snapshots intermediários (mantém os 2 últimos)
├── splat.ply           resultado final
└── export/             exportações otimizadas (.sog / .ply)
```

### Exportação otimizada

Com o splat treinado, o painel **Exportação** gera uma versão menor sem
retreinar (dá para testar ajustes e alternar **Ver no 3D / Ver original**):

| Ajuste | O que faz |
|---|---|
| **SOG comprimido** | Formato do PlayCanvas (v2): texturas WebP sem perdas com posições em 16 bits, quatérnios "smallest three", escalas e cores em codebooks de 256 e as bandas SH numa paleta k-means de até 64k vetores. Abre no Spark (o visualizador do app), PlayCanvas e SuperSplat. |
| **Reduzir** | Funde gaussianas parecidas (vizinhas, de forma e cor próximas) até remover a porcentagem pedida. A opacidade conserva a "tinta" (opacidade × área), então as superfícies não abrem buracos. |
| **Opacidade mín.** | Descarta splats quase invisíveis. |
| **Bandas SH** | Corta o brilho dependente do ângulo; cada grau a menos encolhe bastante o .ply. |

Port do que o WebGaussian faz no navegador, em Rust com rayon. Medido num M4:

| Entrada | Saída | Tempo |
|---|---|---|
| 111k splats, SH 1, 10,3 MB | `.sog` 2,1 MB (4,9×) | 2,5 s |
| idem, −50%, opac. ≥ 2%, SH 0 | `.sog` 0,69 MB (14,9×) | 0,2 s |
| 1M splats, SH 3, 236 MB | `.sog` 18 MB (12,8×) | 15,7 s |
| idem, −50% | `.ply` 118 MB | 1,2 s |

O `.sog` foi conferido com o leitor de referência (`splat-transform` da
PlayCanvas): erro de posição de 5×10⁻⁵ da extensão da cena, rotação abaixo de
1,2° (p99) e coeficientes SH com erro mediano de 0,001. O WebP sai do libwebp
(compilado do fonte pelo `libwebp-sys`), ~18% menor que o encoder em Rust puro.

### Linha de comando (lote)

O mesmo pipeline roda sem interface, com eventos em JSON por linha:

```bash
cd src-tauri
cargo run --release --example cli -- config.json      # formato de JobConfig
cargo run --release --example cli -- scene <projeto>  # enquadramento da cena
cargo run --release --example cli -- export <projeto> [opcoes.json]
```

`opcoes.json` segue o `ExportOptions`, por exemplo
`{"format": "sog", "decimate": 0.5, "minOpacity": 0.02, "shDegree": 0}`.

## Por que esse pipeline

| Escolha | Motivo |
|---|---|
| Decodificação com VideoToolbox e seleção por nitidez | Extrai N×K candidatos e fica com o mais nítido de cada janela, uniformemente distribuídos na órbita. O cálculo roda em Rust com rayon, em paralelo. |
| Câmera única pinhole | Vídeo gerado não tem distorção de lente; menos parâmetros deixam a calibração mais estável. |
| Matching circular | Cada frame é casado com seus vizinhos, com wrap-around para fechar a volta, mais saltos longos (16/32/64). Sai muito mais barato que o matching exaustivo e não depende de vocab tree. |
| `view_graph_calibrator` + `global_mapper` | O mapeador global do COLMAP 4 (ex-GLOMAP) é bem mais rápido que o incremental. O incremental só entra como fallback, quando o global registra menos de 60% das câmeras. |
| SIFT reforçado | Limiar de detecção baixo (0,002), agrupamento de escalas e forma afim, com 20 vizinhos casados. Num vídeo real de IA com a volta por trás de uma cabeça (cabelo liso, fundo desfocado), o SIFT padrão registrou 83/107 câmeras; o reforçado registrou 107/107 em ~110 s. Com o SIFT padrão, se ficar abaixo de 95% das câmeras, o app repete automaticamente com o reforçado. |
| ALIKED + LightGlue (opcional) | Também registrou 107/107 nesse vídeo, mas levou 16,5 min na CPU e teve erro de reprojeção maior (1,5 px contra 0,84 px). |
| Brush como sidecar | Treino na GPU via Metal, sem CUDA. Exporta snapshots que o visualizador (Spark/three.js) recarrega ao vivo; o visualizador só renderiza quando algo muda, para não disputar a GPU com o treino. |
| Perda SSIM desligada por padrão | No Metal ela triplica o custo de cada passo (8,8 → 26,7 passos/s sem ela). Num vídeo real (107 frames, 960×1280), sem SSIM 15k passos deram 32,6 dB em 8,6 min; com SSIM, 7k passos deram 31,8 dB em 9,3 min e 20k deram 33,2 dB em ~28 min. O preset Qualidade mantém SSIM. |
| Trava de ângulo no visualizador | A órbita da câmera fica na faixa de alturas que o vídeo filmou (+20°), onde há dados. |
| PSNR em frames separados | Mede a consistência 3D do vídeo: abaixo de ~25 dB, o vídeo de IA provavelmente não é geometricamente coerente. |

Medição num vídeo sintético de teste (960×540, 120 frames, M4 16 GB):
frames em 0,6 s; poses em 25 s (120/120 câmeras, 0,57 px); 3 mil passos
de treino em 94 s, chegando a PSNR 35,7 dB.

## Desenvolvimento da interface sem o backend

`pnpm dev:mock` sobe só a interface no navegador e reproduz eventos gravados
pela CLI. Configure em `.env.local`:

```
VITE_MOCK_VIDEO=/caminho/video.mp4
VITE_MOCK_EVENTS=/caminho/eventos.jsonl   # stdout da CLI
VITE_MOCK_SCENE=/caminho/scene.json       # saída de `cli scene`
```

Na URL, `?os=windows` simula a interface do Windows e `?events=…&scene=…`
troca os arquivos sem editar o `.env.local`. `?project=<pasta>` abre um projeto
real; a exportação é simulada e devolve o `lastExport` gravado nele por
`cli export`. Use uma pasta fora de `~/Downloads`/`~/Documents`: o macOS pode
bloquear o acesso do Vite a elas e o servidor trava.

## Protótipo web (treino no navegador)

`web/` é um protótipo que roda **só o treino** no navegador: o Brush compilado
para WebAssembly (bindings `brush-js`) treina na GPU via WebGPU, com prévias
no mesmo visualizador Spark do app e download do `.ply` no fim. Frames,
recorte e poses continuam no app desktop (ou em qualquer ferramenta que
exporte COLMAP).

```bash
pnpm web:wasm           # clona o Brush num commit fixo, aplica web/brush/*.patch e compila (~5 min)
pnpm web:dev            # http://localhost:1440
```

O `web:wasm` precisa de `rustup` e `wasm-pack`; instala sozinho um toolchain
Rust 1.99 lado a lado (o Burn do Brush exige ≥ 1.95) sem mexer no `stable`.

- **Entrada:** a pasta `dataset` (ou `dataset_alpha`) de um projeto do
  SplatForge, ou qualquer export COLMAP com `images/` e `sparse/0`. No Chrome
  e no Edge a pasta é lida direto (`showDirectoryPicker`); nos outros
  navegadores é copiada para o OPFS. `?dataset=<url>` baixa um dataset servido
  por HTTP (os nomes das imagens vêm do `images.bin`); em dev,
  `WEB_FS_ANY=1 pnpm web:dev` e `?dataset=/@fs/<caminho absoluto>`.
- **Navegador:** Chrome/Edge desktop com WebGPU e a extensão `subgroups`.
- **Desempenho (M4, dataset sintético de 72 imagens 480×360, 3 mil passos,
  SH 1):** 38 s no Chrome (~100 passos/s), contra 12 s do Brush nativo; PSNR
  27,8 dB (nativo v0.3.0: 26,2 dB). O `.ply` sai no mesmo formato do export nativo.

O que foi preciso consertar para o treino rodar (e rodar rápido) no navegador:

- **Commit do Brush:** fixado antes do #546, que leva o kernel
  `project_forward` a 12 storage buffers; o Chrome expõe só 10 por estágio
  no Metal e o treino não roda no Mac.
- **`enable subgroups;`:** o writer WGSL do naga (wgpu 30) usa as funções de
  subgroup sem declarar a extensão, e o Chrome recusa os shaders. O
  `web/src/brush.ts` acrescenta a diretiva antes de entregar o WGSL ao
  navegador.
- **Patch `web/brush/brush-js.patch`:**
  - Construtores estáticos rodam uma vez só. Sem isso, o wasm-ld trata o
    módulo como "command" e embrulha **toda** export (até `malloc`) para
    rodar de novo os milhares de registros `inventory` do CubeCL a cada
    chamada JS→WASM. O treino ficava ~25× mais lento (4 passos/s).
  - Autotune do CubeCL em `Full`: nos outros níveis o Burn mede a vazão de
    pico da GPU com um `sync` bloqueante, que dá panic no WASM no primeiro
    reduce.
  - `trainSteps` não consulta de novo um stream já encerrado (panic "async fn
    resumed after completion" no fim do treino).
  - `Training.exportPly()`: o mesmo export do Brush nativo (piso de escala
    embutido, unidades do dataset restauradas).

### Vídeo → splat no navegador

O painel **Vídeo** faz no navegador as etapas que faltavam, com a mesma lógica
do app desktop, num projeto no OPFS (`dataset/`, `masks/`, `dataset_alpha/`):

1. **Frames:** o navegador decodifica o vídeo (um seek por candidato).
   Candidatos uniformes no trecho (`frames × candidatos`, no máximo o fps do
   vídeo), escala que cabe no lado máximo sem ampliar, e o mais nítido de cada
   janela (variância do Laplaciano na miniatura de 640 px, luma Rec.709), como
   no `frames.rs`. O seek vai meio quadro adiante para pegar o quadro mais
   próximo do instante, como o filtro `fps` do ffmpeg.
2. **Recorte:**
   - **IA (RMBG-1.4)**: 176 MB, licença não comercial, só com aceite. Roda no
     ONNX Runtime Web com WebGPU, com o mesmo pré/pós-processamento do
     `masks.rs`. O modelo vem do mesmo endereço fixado do instalador, com
     SHA-256 conferido, e fica em cache no OPFS. É o único modelo de IA da
     versão web: o Vision é só do macOS, o BiRefNet é lento até em GPU
     dedicada e o MODNet (pessoas) ficou de fora por decisão do projeto.
   - **Cor sólida** e **despill**: port exato do Rust (testes em
     `web/src/pipeline/*.test.ts`, `pnpm test:web`, com os casos do
     `masks.rs`).
   - `dataset_alpha`: PNG RGBA com alfa direto (encoder próprio: o canvas
     pré-multiplica o alfa) e poses com os nomes `.png`. "Usar o recorte
     também nas poses" descarta as features onde a máscara é 0.
3. **Poses:** o SfM descrito abaixo, já com o recorte se pedido.

Medido num vídeo de tela verde sintético (`synth-orbit.py --green`: 120
quadros 960×720, 24 fps, verde rebatido nos objetos), preset Prévia (90
frames), no Chrome headless do M4:

| Etapa | Tempo | Resultado |
|---|---|---|
| Frames | 3,1 s | 90 de 120 candidatos |
| Recorte, cor sólida | 2,4 s | IoU 0,91 com as máscaras verdadeiras (mín. 0,83) |
| Recorte, RMBG-1.4 (WebGPU) | 32 s (+12 s no 1º uso: download e verificação) | IoU 0,996 (mín. 0,99), ~0,36 s/frame |
| Poses | 29–31 s | 90/90 câmeras, posição 0,15% do raio, rotação 0,12°, focal −0,4% |
| PNG com transparência | 4,5 s | pixels do sujeito com verde dominante: 16% → 0% |
| Treino (3 mil passos, cor sólida) | — | PSNR 36,7 dB, 0,4% das gaussianas visíveis verdes |

O chroma key não chega a IoU 1 porque o cubo tem manchas verde-azuladas que
encostam na borda e saem junto com a tela (o mesmo acontece no app desktop);
o RMBG + despill as mantém. Num painel ou aba oculta o navegador limita os
seeks e a extração de frames fica ~10× mais lenta.

### Poses no navegador (prova de conceito)

Com uma pasta que tenha `images/`, o botão **"Calcular poses no navegador"**
faz o SfM sem o COLMAP, numa cópia das imagens no OPFS (a pasta original não
é alterada), e grava um `sparse/0` que o treino usa em seguida:

1. **Features:** AKAZE do OpenCV.js (`@techstark/opencv-js` 4.12, que não
   traz SIFT nem as funções de geometria epipolar) em Web Workers.
2. **Matching:** força bruta por distância de Hamming num compute shader
   WebGPU, com teste de razão e checagem mútua; mesmos pares do app (janela do
   preset + saltos 16/32/64, órbita fechada).
3. **Verificação:** matriz fundamental por RANSAC (8 pontos normalizado).
4. **Reconstrução incremental** num worker (`web/src/sfm/`): tracks por
   union-find, par inicial escolhido por paralaxe (≥ 2°) e número de pontos,
   com até 3 tentativas em regiões diferentes da órbita se a primeira
   registrar menos de 90% das câmeras, PnP partindo da pose do frame
   vizinho (com DLT + RANSAC de reserva), triangulação, BA local a cada frame
   e global periódico (Levenberg–Marquardt com complemento de Schur, Huber,
   focal compartilhada). No fim, BA com limiar decrescente reaproveita as
   observações que fecham a órbita.

Medido no M4 com um vídeo orbital sintético de 120 frames 960×720 e poses
verdadeiras conhecidas (`synth2.py`: textura procedural, chão e parede ao
fundo):

| | Tempo | Câmeras | Pontos | Focal | Posição (mediana) | Rotação (mediana) | PSNR do treino |
|---|---|---|---|---|---|---|---|
| Navegador | 37 s | 120/120 | 14 mil | +0,22% | 0,27% do raio | 0,085° | 39,7 dB |
| COLMAP 4.2 nativo (SIFT, global) | 14 s | 120/120 | 2 mil | +0,07% | 0,73% do raio | 0,19° | 36,2 dB |
| Poses verdadeiras | — | — | — | — | — | — | 41,2 dB |

PSNR: Brush web, 5 mil passos a 960 px, SH 1, 8 vistas de avaliação.

Para repetir a medição (`scripts/web-bench/`, com `WEB_FS_ANY=1 pnpm web:dev`
rodando):

```bash
python3 scripts/web-bench/synth-orbit.py /tmp/orbit 120 960 720
```
```bash
node scripts/web-bench/headless.mjs poses "http://localhost:1440/?dataset=/@fs/tmp/orbit" --out /tmp/orbit-web
```
```bash
python3 scripts/web-bench/eval-poses.py /tmp/orbit/gt.json /tmp/orbit-web/sparse/0
```
```bash
node scripts/web-bench/headless.mjs train "http://localhost:1440/?dataset=/@fs/tmp/orbit-web" --steps 5000 --max-res 960
```

O `synth-orbit.py` gera um dataset já com as poses verdadeiras em `sparse/0`
(treino de referência) e o `gt.json`; o `headless.mjs` roda o protótipo num
Chrome headless (sem o throttling de abas ocultas) nos modos `poses`, `train`
e `profile` (perfil de CPU do treino; com `pnpm web:wasm --profiling` sai com
os nomes das funções do WASM).

Ainda não foi testado em vídeos reais de IA (desfoque, pouca textura,
elementos que se mexem), onde o SIFT reforçado e o mapeador global do COLMAP
fazem diferença.

## Testes

```bash
pnpm test:rust
```
