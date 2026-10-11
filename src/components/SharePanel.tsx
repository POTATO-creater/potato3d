import { useEffect, useRef, useState } from "react";
import { host, type SceneFrame } from "../lib/api";
import { fmtBytes } from "../lib/format";
import { basename } from "../lib/platform";
import {
  revokeShare,
  saveShare,
  savedShare,
  ShareError,
  shareKey,
  shareView,
  uploadShare,
  type ShareLink,
} from "../lib/share";
import { useI18n } from "../i18n";

interface Props {
  /** Arquivo exportado que vai para o link. */
  path: string;
  bytes: number;
  /** Enquadramento do vídeo, para o link abrir no mesmo ponto de vista. */
  frame: SceneFrame | null;
  onClose: () => void;
}

/** Link privado e temporário para a exportação (src/lib/share.ts). Aparece
 *  abaixo do resultado da exportação, aberto pelo botão "Compartilhar". */
export function SharePanel({ path, bytes, frame, onClose }: Props) {
  const { t, locale } = useI18n();
  const ts = t.share;
  const key = shareKey(path, bytes);
  const [link, setLink] = useState<ShareLink | null>(() => savedShare(key));
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const upload = useRef<AbortController | null>(null);

  useEffect(() => {
    setLink(savedShare(key));
    setError(null);
    return () => upload.current?.abort();
  }, [key]);

  // O link some daqui quando expira.
  useEffect(() => {
    if (!link) return;
    const timer = setTimeout(() => setLink(null), Math.min(link.expiresAt - Date.now(), 2 ** 31 - 1));
    return () => clearTimeout(timer);
  }, [link]);

  function describe(e: unknown) {
    if (!(e instanceof ShareError)) return String(e);
    const errors = ts.errors as Record<string, string | ((max: string) => string)>;
    const msg = errors[e.code] ?? ts.errors.server;
    return typeof msg === "function" ? msg(e.maxBytes ? fmtBytes(e.maxBytes) : "?") : msg;
  }

  async function create() {
    const ctl = new AbortController();
    upload.current = ctl;
    setError(null);
    setProgress(0);
    try {
      const file = await host.readBlob(path);
      const made = await uploadShare(host.shareOrigin, file, basename(path), shareView(frame), setProgress, ctl.signal);
      saveShare(key, made);
      setLink(made);
    } catch (e) {
      if (!ctl.signal.aborted) setError(describe(e));
    } finally {
      if (upload.current === ctl) upload.current = null;
      setProgress(null);
    }
  }

  async function revoke() {
    if (!link) return;
    setError(null);
    try {
      await revokeShare(host.shareOrigin, link);
      saveShare(key, null);
      setLink(null);
      onClose();
    } catch (e) {
      setError(describe(e));
    }
  }

  async function copy() {
    if (!link) return;
    try {
      await navigator.clipboard.writeText(link.url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* sem permissão: o campo está selecionável */
    }
  }

  const expires = link ? new Date(link.expiresAt) : null;
  const sameDay = expires?.toDateString() === new Date().toDateString();
  const when = expires?.toLocaleString(locale, {
    ...(sameDay ? {} : { day: "2-digit", month: "2-digit" }),
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="share">
      {link ? (
        <>
          <div className="share-link">
            <input
              className="share-url mono"
              readOnly
              value={link.url}
              onFocus={(e) => e.currentTarget.select()}
              aria-label={ts.button}
            />
            <button className="btn-sm" onClick={copy}>
              {copied ? ts.copied : ts.copy}
            </button>
          </div>
          <div className="share-body">
            <QrCode text={link.url} />
            <div className="share-info">
              <p className="help">{ts.expires(when ?? "")}</p>
              <p className="help">{ts.qrHint}</p>
              <div className="trim-actions">
                <a className="btn-sm ghost share-btn-link" href={link.url} target="_blank" rel="noreferrer">
                  {ts.open}
                </a>
                <button className="btn-sm ghost" onClick={revoke} title={ts.revokeTitle}>
                  {ts.revoke}
                </button>
              </div>
            </div>
          </div>
        </>
      ) : progress != null ? (
        <div className="export-progress">
          <div className="bar">
            <div className="bar-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
          </div>
          <div className="share-uploading">
            <span className="help">{ts.uploading(Math.round(progress * 100))}</span>
            <button className="btn-sm ghost" onClick={() => upload.current?.abort()}>
              {t.common.cancel}
            </button>
          </div>
        </div>
      ) : (
        <>
          <p className="help">{ts.intro(fmtBytes(bytes))}</p>
          <div className="trim-actions">
            <button className="btn-sm" onClick={create}>
              {ts.create}
            </button>
            <button className="btn-sm ghost" onClick={onClose}>
              {t.common.cancel}
            </button>
          </div>
        </>
      )}
      {error && <p className="help export-error">{error}</p>}
    </div>
  );
}

/** QR code do link, para abrir no celular. A biblioteca só carrega aqui. */
function QrCode({ text }: { text: string }) {
  const [qr, setQr] = useState<{ d: string; n: number } | null>(null);
  useEffect(() => {
    let alive = true;
    import("qrcode-generator").then(({ default: qrcode }) => {
      const code = qrcode(0, "M");
      code.addData(text);
      code.make();
      const n = code.getModuleCount();
      let d = "";
      for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (code.isDark(y, x)) d += `M${x} ${y}h1v1h-1z`;
      if (alive) setQr({ d, n });
    });
    return () => {
      alive = false;
    };
  }, [text]);
  if (!qr) return <div className="share-qr" />;
  // Margem de 2 módulos e fundo branco: leitores de QR precisam do contraste.
  return (
    <svg className="share-qr" viewBox={`-2 -2 ${qr.n + 4} ${qr.n + 4}`} shapeRendering="crispEdges" role="img">
      <rect x={-2} y={-2} width={qr.n + 4} height={qr.n + 4} fill="#fff" />
      <path d={qr.d} fill="#000" />
    </svg>
  );
}
