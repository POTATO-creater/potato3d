import type { ChromaOptions, MaskMethod, MaskOptions } from "../lib/api";
import type { Os } from "../lib/platform";
import { Segmented, Slider, Toggle } from "./fields";
import { useI18n, type Messages } from "../i18n";

const fmtTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

const toHex = (c: [number, number, number]) => `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
const fromHex = (h: string): [number, number, number] => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as never;

/** Métodos de recorte disponíveis em cada sistema. */
function methods(t: Messages, os: Os | undefined): { value: MaskMethod; label: string; hint: string }[] {
  const ids: MaskMethod[] =
    os === "windows"
      ? ["people", "general", "rmbg", "chroma"]
      : os === "web"
        ? ["rmbg", "sam2", "chroma"]
        : os === "macos"
          ? ["auto", "rmbg", "chroma"]
          : ["auto", "chroma"];
  return ids.map((value) => ({ value, ...t.masks.methods[value] }));
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
  const { t } = useI18n();
  const tm = t.masks;
  const options = methods(t, os);
  // Método salvo pode não existir neste sistema (ex.: "people" vindo do Windows).
  const method = options.some((o) => o.value === masks.method) ? masks.method : options[0].value;
  const c = masks.chroma;
  const setChroma = (patch: Partial<ChromaOptions>) => onMasks({ ...masks, method, chroma: { ...c, ...patch } });

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{tm.title}</h2>
      </header>
      <Toggle
        label={tm.enable}
        hint={tm.enableHint}
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

          {method === "sam2" && (
            <div className="sam2-prompt">
              {masks.prompt ? (
                <p className="help">
                  {tm.prompt.marked(
                    fmtTime(masks.prompt.time ?? 0),
                    [masks.prompt.box && tm.prompt.box, masks.prompt.points.length && tm.prompt.points(masks.prompt.points.length)]
                      .filter(Boolean)
                      .join(" + "),
                  )}{" "}
                  <button className="link" disabled={locked} onClick={() => onMasks({ ...masks, method, prompt: null })}>
                    {tm.prompt.clear}
                  </button>
                </p>
              ) : (
                <p className="help">{tm.prompt.none}</p>
              )}
              <p className="help muted">{tm.prompt.howTo}</p>
            </div>
          )}

          {method === "chroma" && (
            <fieldset className="chroma" disabled={locked}>
              <div className="chroma-color">
                <label className={`swatch ${c.autoColor ? "auto" : ""}`} title={tm.bgColor}>
                  <input
                    type="color"
                    value={toHex(c.color)}
                    onChange={(e) => setChroma({ color: fromHex(e.currentTarget.value), autoColor: false })}
                  />
                  <span style={{ background: toHex(c.color) }} />
                </label>
                <Toggle
                  label={tm.autoColor}
                  hint={tm.autoColorHint}
                  checked={c.autoColor}
                  onChange={(v) => setChroma({ autoColor: v })}
                />
              </div>
              <Slider label={tm.tolerance} value={c.tolerance} min={0} max={60} onChange={(v) => setChroma({ tolerance: v })} />
              <Slider label={tm.softness} value={c.softness} min={0} max={40} onChange={(v) => setChroma({ softness: v })} />
              <Toggle
                label={tm.contiguous}
                hint={tm.contiguousHint}
                checked={c.contiguous}
                onChange={(v) => setChroma({ contiguous: v })}
              />
            </fieldset>
          )}

          <Toggle
            label={tm.despill}
            hint={tm.despillHint}
            checked={masks.despill}
            disabled={locked}
            onChange={(v) => onMasks({ ...masks, method, despill: v })}
          />
          {masks.despill && (
            <fieldset className="chroma" disabled={locked}>
              <Slider
                label={tm.despillStrength}
                value={masks.despillStrength}
                min={0}
                max={100}
                onChange={(v) => onMasks({ ...masks, method, despillStrength: v })}
              />
            </fieldset>
          )}

          <Toggle
            label={tm.forPoses}
            hint={tm.forPosesHint}
            checked={masks.forPoses}
            disabled={locked}
            onChange={(v) => onMasks({ ...masks, method, forPoses: v })}
          />
        </div>
      )}
    </section>
  );
}
