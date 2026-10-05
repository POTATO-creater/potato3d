# SplatForge

App desktop (Tauri 2 + React) que transforma **vídeos orbitais** — inclusive
gerados por IA — em **Gaussian Splatting**, 100% local, no macOS (Apple
Silicon) e no Windows (com aceleração CUDA no COLMAP quando há GPU NVIDIA).

```
vídeo ──► Frames ──────────► Poses ───────────────────► Treino ──────────► splat.ply
          ffmpeg             COLMAP 4                     Brush
          decodificação      SIFT reforçado,              wgpu (Metal,
          por hardware +     matching circular,           Vulkan/DX12),
          nitidez (Rust)     mapeador global              prévias ao vivo
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

Cada vídeo ganha uma pasta de projeto (`<video>_splat/` por padrão). Para
mudar só os parâmetros de treino, escolha **Executar a partir de → Treino**.
Assim frames e poses são reaproveitados.

```
<video>_splat/
├── project.json        estado (frames, métricas, resultado)
├── dataset/            entrada do Brush: images/ + sparse/0/
├── colmap/             database.db, pairs.txt, modelos brutos
├── training/           snapshots intermediários (mantém os 2 últimos)
└── splat.ply           resultado final
```

### Linha de comando (lote)

O mesmo pipeline roda sem interface, com eventos em JSON por linha:

```bash
cd src-tauri
cargo run --release --example cli -- config.json      # formato de JobConfig
cargo run --release --example cli -- scene <projeto>  # enquadramento da cena
```

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
troca os arquivos sem editar o `.env.local`.

## Testes

```bash
pnpm test:rust
```
