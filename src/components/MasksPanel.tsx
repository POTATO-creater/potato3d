import { useState } from "react";
import type { ChromaOptions, MaskMethod, MaskOptions, MaskSource, MaskSourceOf } from "../lib/api";
import { Segmented, Slider, Toggle } from "./fields";
import { MaskPreview, type PreviewSource } from "./MaskPreview";
import { MaskSourceBox } from "./MaskSourceBox";
import { useI18n, type Messages } from "../i18n";

const fmtTime = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

const toHex = (c: [number, number, number]) => `#${c.map((v) => v.toString(16).padStart(2, "0")).join("")}`;
const fromHex = (h: string): [number, number, number] => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as never;

export const MASK_METHODS: MaskMethod[] = ["rmbg", "sam2", "chroma", "external"];

function methods(t: Messages): { value: MaskMethod; label: string; hint: string }[] {
  return MASK_METHODS.map((value) => ({ value, ...t.masks.methods[value] }));
}

export function MasksPanel({
  masks,
  onMasks,
  onApplyAll,
  locked,
  previewPaused,
  preview,
  promptOutside,
  projectDir,
  source,
  sourceSize,
  onMaskSource,
}: {
  masks: MaskOptions;
  onMasks: (m: MaskOptions) => void;
  /** Batch: copies this masking to the other items. */
  onApplyAll?: () => void;
  locked: boolean;
  /** The preview waits while any job holds the GPU. */
  previewPaused?: boolean;
  /** Frame for the mask preview (where the host provides a preview). */
  preview: PreviewSource | null;
  /** The marking fell outside the trim range (it will be ignored). */
  promptOutside: boolean;
  /** Method "external": the item's project and source, which the mask
   *  video or frames are matched to. */
  projectDir: string | null;
  source: MaskSourceOf | null;
  sourceSize: [number, number] | null;
  onMaskSource: (projectDir: string, src: MaskSource | null) => void;
}) {
  const { t } = useI18n();
  const tm = t.masks;
  const options = methods(t);
  // The saved method may not exist on this OS (e.g. "people" from Windows).
  const method = options.some((o) => o.value === masks.method) ? masks.method : options[0].value;
  const c = masks.chroma;
  const setChroma = (patch: Partial<ChromaOptions>) => onMasks({ ...masks, method, chroma: { ...c, ...patch } });
  // What is attached, for the preview.
  const [attached, setAttached] = useState<MaskSource | null>(null);
  const external = method === "external";

  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{tm.title}</h2>
        {onApplyAll && (
          <span className="panel-actions">
            <button className="link" onClick={onApplyAll} title={tm.applyAllTitle}>
              {tm.applyAll}
            </button>
          </span>
        )}
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
                    masks.prompt.image ?? fmtTime(masks.prompt.time ?? 0),
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
              {promptOutside && <p className="help error">{tm.prompt.outside}</p>}
              <p className="help muted">{tm.prompt.howTo}</p>
            </div>
          )}

          {external && (
            <MaskSourceBox
              projectDir={projectDir}
              source={source}
              sourceSize={sourceSize}
              locked={locked}
              onChanged={onMaskSource}
              onCurrent={setAttached}
            />
          )}

          {/* As in the job: a marking outside the trim range doesn't apply. */}
          {(!external || (attached && projectDir)) && (
            <MaskPreview
              source={preview}
              masks={{ ...masks, method, prompt: promptOutside ? null : masks.prompt }}
              locked={locked || !!previewPaused}
              external={external && attached && projectDir ? { projectDir, id: attached.id } : null}
            />
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

          {method === "chroma" && (
            <Toggle
              label={tm.despill}
              hint={tm.despillHint}
              checked={masks.despill}
              disabled={locked}
              onChange={(v) => onMasks({ ...masks, method, despill: v })}
            />
          )}
          {method === "chroma" && masks.despill && (
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
