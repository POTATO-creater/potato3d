import type { ChromaOptions, MaskMethod, MaskOptions } from "../lib/api";
import type { Os } from "../lib/platform";
import { Segmented, Slider, Toggle } from "./fields";

const toHex = (c: [number, number, number]) => `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
const fromHex = (h: string): [number, number, number] => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as never;

/** Métodos de recorte disponíveis em cada sistema. */
function methods(os: Os | undefined): { value: MaskMethod; label: string; hint: string }[] {
  const chroma = {
    value: "chroma" as const,
    label: "Cor sólida",
    hint: "Chroma key: instantâneo. Para vídeos com fundo de cor única (tela verde, azul…).",
  };
  if (os === "windows") {
    return [
      { value: "people", label: "Pessoas", hint: "MODNet na GPU: rápido, mas só reconhece pessoas." },
      { value: "general", label: "Objetos", hint: "BiRefNet: qualquer sujeito, bem mais lento." },
      chroma,
    ];
  }
  return [{ value: "auto", label: "IA (Vision)", hint: "Recorte nativo do macOS, na GPU: qualquer sujeito." }, chroma];
}

export function MasksPanel({
  os,
  masks,
  onMasks,
  locked,
}: {
  os: Os | undefined;
  masks: MaskOptions;
  onMasks: (m: MaskOptions) => void;
  locked: boolean;
}) {
  const options = methods(os);
  // Método salvo pode não existir neste sistema (ex.: "people" vindo do Windows).
  const method = options.some((o) => o.value === masks.method) ? masks.method : options[0].value;
  const c = masks.chroma;
  const setChroma = (patch: Partial<ChromaOptions>) => onMasks({ ...masks, method, chroma: { ...c, ...patch } });

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>Recorte do sujeito</h2>
      </header>
      <Toggle
        label="Isolar o sujeito (remover o fundo)"
        hint="o splat sai só com a pessoa ou objeto, sem paredes e riscos em volta"
        checked={masks.enabled}
        disabled={locked}
        onChange={(v) => onMasks({ ...masks, enabled: v, method })}
      />
      {masks.enabled && (
        <div className="masks-body">
          <Segmented
            value={method}
            disabled={locked}
            onChange={(v) => onMasks({ ...masks, method: v })}
            options={options.map((o) => ({ value: o.value, label: o.label, title: o.hint }))}
          />
          <p className="help">{options.find((o) => o.value === method)?.hint}</p>

          {method === "chroma" && (
            <fieldset className="chroma" disabled={locked}>
              <div className="chroma-color">
                <label className={`swatch ${c.autoColor ? "auto" : ""}`} title="Cor do fundo">
                  <input
                    type="color"
                    value={toHex(c.color)}
                    onChange={(e) => setChroma({ color: fromHex(e.currentTarget.value), autoColor: false })}
                  />
                  <span style={{ background: toHex(c.color) }} />
                </label>
                <Toggle
                  label="Detectar a cor pelas bordas"
                  hint="mediana das bordas de cada frame"
                  checked={c.autoColor}
                  onChange={(v) => setChroma({ autoColor: v })}
                />
              </div>
              <Slider label="Tolerância" value={c.tolerance} min={0} max={60} onChange={(v) => setChroma({ tolerance: v })} />
              <Slider label="Suavização da borda" value={c.softness} min={0} max={40} onChange={(v) => setChroma({ softness: v })} />
              <Toggle
                label="Só regiões ligadas à borda"
                hint="não fura partes do sujeito com a mesma cor do fundo"
                checked={c.contiguous}
                onChange={(v) => setChroma({ contiguous: v })}
              />
              <Toggle
                label="Remover reflexo da cor (despill)"
                hint="tira a franja colorida na borda do sujeito"
                checked={c.despill}
                onChange={(v) => setChroma({ despill: v })}
              />
            </fieldset>
          )}

          <Toggle
            label="Usar o recorte também nas poses"
            hint="ajuda se o fundo muda entre frames; atrapalha em sujeitos com pouca textura"
            checked={masks.forPoses}
            disabled={locked}
            onChange={(v) => onMasks({ ...masks, method, forPoses: v })}
          />
        </div>
      )}
    </section>
  );
}
