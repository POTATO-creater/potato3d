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

## Testes

```bash
pnpm test:rust
```
