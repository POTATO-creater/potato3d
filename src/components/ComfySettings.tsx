import { useEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../i18n";
import type { Messages } from "../i18n/pt-BR";
import { ComfyClient, ComfyError } from "../lib/comfy/client";
import { loadComfy, saveComfy, type ComfySettings as Settings, type ComfyWorkflow } from "../lib/comfy/settings";
import { checkIO, effectiveIO, fileFields, guessIO, listNodes, numberFields, parseWorkflow, WorkflowParseError, type LengthParam } from "../lib/comfy/workflow";
import { Field, NumField, Select } from "./fields";

/** Texto de um erro do ComfyUI para a interface. */
export function comfyErrorText(t: Messages, e: unknown): string {
  const te = t.comfy.errors;
  if (e instanceof ComfyError) {
    switch (e.reason) {
      case "network":
        return te.network;
      case "http":
        return te.http(e.detail);
      case "validation":
        return te.validation(e.detail);
      case "execution":
        return te.execution(e.detail);
      case "interrupted":
        return te.interrupted;
      case "noOutput":
        return te.noOutput(e.detail);
    }
  }
  return String((e as Error)?.message ?? e);
}

const ICON_CLOSE = "M4.5 4.5l9 9M13.5 4.5l-9 9";

/** Configuração da integração: endereço do ComfyUI e workflows importados,
 *  com os nós de entrada e saída de cada um. Fica no localStorage. */
export function ComfySettings({ onClose }: { onClose: () => void }) {
  const { t } = useI18n();
  const tc = t.comfy;
  const [settings, setSettings] = useState<Settings>(loadComfy);
  const [saveFailed, setSaveFailed] = useState(false);
  const [test, setTest] = useState<{ state: "idle" | "testing" | "ok" | "error"; text?: string }>({ state: "idle" });
  const [importError, setImportError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const update = (s: Settings) => {
    setSettings(s);
    setSaveFailed(!saveComfy(s));
  };
  const updateWorkflow = (id: string, patch: Partial<ComfyWorkflow>) =>
    update({ ...settings, workflows: settings.workflows.map((w) => (w.id === id ? { ...w, ...patch } : w)) });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Um Select aberto consome o Esc (preventDefault) para fechar a lista.
      if (e.key !== "Escape" || e.defaultPrevented) return;
      onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const runTest = async () => {
    setTest({ state: "testing" });
    try {
      const s = await new ComfyClient(settings.url).stats(AbortSignal.timeout(8000));
      const what = [s.version && `ComfyUI ${s.version}`, s.device?.replace(/^cuda:\d+\s*/, ""), s.vramGb && `${s.vramGb.toFixed(0)} GB`]
        .filter(Boolean)
        .join(" · ");
      setTest({ state: "ok", text: tc.connected(what || settings.url) });
    } catch (e) {
      setTest({ state: "error", text: (e as Error).name === "TimeoutError" ? tc.errors.network : comfyErrorText(t, e) });
    }
  };

  const importFile = async (file: File) => {
    setImportError(null);
    try {
      const graph = parseWorkflow(await file.text());
      const io = guessIO(graph) ?? { input: { node: "", field: "" }, output: { node: "" } };
      const wf: ComfyWorkflow = {
        id: crypto.randomUUID(),
        name: file.name.replace(/\.json$/i, ""),
        kind: "restore",
        graph,
        io,
        importedAt: Date.now(),
      };
      update({ ...settings, workflows: [...settings.workflows, wf] });
    } catch (e) {
      setImportError(e instanceof WorkflowParseError ? tc.parseErrors[e.reason] : String(e));
    }
  };

  return (
    <div className="modal" role="dialog" aria-modal="true" aria-labelledby="comfy-title" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-card comfy-settings">
        <header className="modal-head">
          <h2 id="comfy-title">{tc.title}</h2>
          <button className="modal-close" onClick={onClose} aria-label={tc.close} title={`${tc.close} (Esc)`}>
            <svg viewBox="0 0 18 18" aria-hidden>
              <path d={ICON_CLOSE} />
            </svg>
          </button>
        </header>

        <section className="modal-section">
          <h3>{tc.server}</h3>
          <Field label={tc.url} hint={tc.urlHint}>
            <span className="comfy-url-row">
              <input
                className="input mono"
                value={settings.url}
                spellCheck={false}
                placeholder="http://127.0.0.1:8188"
                onChange={(e) => {
                  update({ ...settings, url: e.currentTarget.value });
                  setTest({ state: "idle" });
                }}
              />
              <button className="btn-sm" onClick={runTest} disabled={test.state === "testing" || !settings.url.trim()}>
                {test.state === "testing" ? tc.testing : tc.test}
              </button>
            </span>
          </Field>
          {test.text && <p className={`comfy-test ${test.state === "ok" ? "good" : "bad"}`}>{test.text}</p>}
          <p className="help">{tc.corsHelp}</p>
        </section>

        <section className="modal-section">
          <div className="modal-section-head">
            <h3>{tc.workflows}</h3>
            <button className="btn-sm" onClick={() => fileInput.current?.click()}>
              {tc.import}
            </button>
            <input
              ref={fileInput}
              type="file"
              accept=".json,application/json"
              hidden
              onChange={(e) => {
                const f = e.currentTarget.files?.[0];
                e.currentTarget.value = "";
                if (f) void importFile(f);
              }}
            />
          </div>
          {importError && <p className="bad">{importError}</p>}
          {settings.workflows.length === 0 && <p className="help">{tc.empty}</p>}
          {settings.workflows.map((w) => (
            <WorkflowCard
              key={w.id}
              wf={w}
              onChange={(patch) => updateWorkflow(w.id, patch)}
              onRemove={() => update({ ...settings, workflows: settings.workflows.filter((x) => x.id !== w.id) })}
            />
          ))}
        </section>
        {saveFailed && <p className="bad">{tc.storageFull}</p>}
      </div>
    </div>
  );
}

function WorkflowCard({ wf, onChange, onRemove }: { wf: ComfyWorkflow; onChange: (p: Partial<ComfyWorkflow>) => void; onRemove: () => void }) {
  const { t } = useI18n();
  const tc = t.comfy;
  const nodes = useMemo(() => listNodes(wf.graph), [wf.graph]);
  const inputNode = nodes.find((n) => n.id === wf.io.input.node);
  const fields = inputNode ? fileFields(inputNode) : [];
  // Duração e fps: o que o usuário escolheu ou, se ainda não, o palpite.
  const io = effectiveIO(wf.graph, wf.io);
  const problems = checkIO(wf.graph, io);
  const length = io.length;
  const lengthNode = length ? nodes.find((n) => n.id === length.node) : undefined;
  const lengthFields = lengthNode ? numberFields(lengthNode) : [];
  const setLength = (l: LengthParam | null) => onChange({ io: { ...wf.io, length: l } });
  const label = (n: (typeof nodes)[number]) => `${n.id} · ${n.title}`;
  // Nós com campo de arquivo primeiro; o resto também vale (nós customizados).
  const inputOptions = [...nodes].sort((a, b) => Number(fileFields(b).length > 0) - Number(fileFields(a).length > 0));
  const outputOptions = [...nodes].sort((a, b) => Number(/save|combine/i.test(b.classType)) - Number(/save|combine/i.test(a.classType)));

  return (
    <div className="comfy-wf">
      <div className="comfy-wf-head">
        <input className="input" value={wf.name} aria-label={tc.name} onChange={(e) => onChange({ name: e.currentTarget.value })} />
        <span className="muted small">{tc.nodes(nodes.length)}</span>
        <button className="btn-sm ghost" onClick={onRemove}>
          {tc.remove}
        </button>
      </div>
      <Field label={tc.kind} hint={tc.kindHints[wf.kind]}>
        <Select value={wf.kind} options={[{ value: "restore" as const, label: tc.kinds.restore }]} onChange={(kind) => onChange({ kind })} />
      </Field>
      {problems.length > 0 && !wf.io.input.node && <p className="warn small">{tc.noGuess}</p>}
      <div className="grid2">
        <Field label={tc.inputNode}>
          <Select
            value={wf.io.input.node}
            aria-label={tc.inputNode}
            options={inputOptions.map((n) => ({ value: n.id, label: label(n), hint: n.classType }))}
            onChange={(node) => {
              const n = nodes.find((x) => x.id === node);
              onChange({ io: { ...wf.io, input: { node, field: n ? (fileFields(n)[0] ?? "") : "" } } });
            }}
          />
        </Field>
        <Field label={tc.inputField}>
          <Select
            value={wf.io.input.field}
            aria-label={tc.inputField}
            disabled={!fields.length}
            options={fields.map((f) => ({ value: f, label: f }))}
            onChange={(field) => onChange({ io: { ...wf.io, input: { ...wf.io.input, field } } })}
          />
        </Field>
      </div>
      <Field label={tc.outputNode}>
        <Select
          value={wf.io.output.node}
          aria-label={tc.outputNode}
          options={outputOptions.map((n) => ({ value: n.id, label: label(n), hint: n.classType }))}
          onChange={(node) => onChange({ io: { ...wf.io, output: { node } } })}
        />
      </Field>
      <h4 className="comfy-wf-sub">{tc.lengthTitle}</h4>
      <p className="help">{tc.lengthHelp}</p>
      <div className="grid2">
        <Field label={tc.lengthNode}>
          <Select
            value={length?.node ?? ""}
            aria-label={tc.lengthNode}
            options={[
              { value: "", label: tc.lengthNone },
              ...nodes.filter((n) => numberFields(n).length > 0).map((n) => ({ value: n.id, label: label(n), hint: n.classType })),
            ]}
            onChange={(node) => {
              const n = nodes.find((x) => x.id === node);
              const field = n ? numberFields(n)[0] : undefined;
              setLength(n && field ? { node, field, unit: length?.unit ?? "seconds" } : null);
            }}
          />
        </Field>
        <Field label={tc.lengthField}>
          <Select
            value={length?.field ?? ""}
            aria-label={tc.lengthField}
            disabled={!length}
            options={lengthFields.map((f) => ({ value: f, label: f, hint: String(lengthNode!.literalInputs.find((i) => i.name === f)?.value) }))}
            onChange={(field) => length && setLength({ ...length, field })}
          />
        </Field>
      </div>
      <div className="grid2">
        <Field label={tc.lengthUnit}>
          <Select
            value={length?.unit ?? "seconds"}
            aria-label={tc.lengthUnit}
            disabled={!length}
            options={[
              { value: "seconds" as const, label: tc.units.seconds },
              { value: "frames" as const, label: tc.units.frames },
            ]}
            onChange={(unit) => length && setLength({ ...length, unit })}
          />
        </Field>
        <NumField label={tc.fps} hint={tc.fpsHint} value={io.fps} min={1} max={120} onChange={(fps) => fps > 0 && onChange({ io: { ...wf.io, fps } })} />
      </div>
      {problems.map((p) => (
        <p key={p} className="bad small">
          {tc.problems[p]}
        </p>
      ))}
    </div>
  );
}

/** Botão do topbar que abre a configuração. */
export function ComfySettingsButton() {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  return (
    <>
      <button className="btn-sm topbar-settings" onClick={() => setOpen(true)} title={t.comfy.openTitle}>
        <svg viewBox="0 0 16 16" aria-hidden>
          <path d="M3 4.5h6M12 4.5h1M3 11.5h1M7 11.5h6" />
          <circle cx="10.5" cy="4.5" r="1.6" />
          <circle cx="5.5" cy="11.5" r="1.6" />
        </svg>
        {t.comfy.open}
      </button>
      {open && <ComfySettings onClose={() => setOpen(false)} />}
    </>
  );
}
