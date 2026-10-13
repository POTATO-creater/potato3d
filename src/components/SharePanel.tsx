import { useEffect, useRef, useState } from "react";
import { host, type SceneFrame } from "../lib/api";
import { fmtBytes } from "../lib/format";
import { basename } from "../lib/platform";
import {
  revokeShare,
  saveShare,
  savedShare,
  setShareAccessKey,
  SHARE_TTLS,
  shareAccessKey,
  shareAdmin,
  ShareError,
  shareKey,
  shareView,
  uploadShare,
  type ShareLink,
  type ShareTtl,
} from "../lib/share";
import { useI18n } from "../i18n";
import { Select } from "./fields";

interface Props {
  /** Exported file to share via the link. */
  path: string;
  bytes: number;
  /** Video framing, so the link opens at the same viewpoint. */
  frame: SceneFrame | null;
  onClose: () => void;
}

/** Private, temporary link for the export (src/lib/share.ts). Shown below the
 *  export result, opened by the "Share" button. With `?admin=true` in the URL
 *  and a saved access key, the lifetime can be chosen. */
export function SharePanel({ path, bytes, frame, onClose }: Props) {
  const { t, locale } = useI18n();
  const ts = t.share;
  const key = shareKey(path, bytes);
  const [link, setLink] = useState<ShareLink | null>(() => savedShare(key));
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [accessKey, setAccessKey] = useState(() => (shareAdmin ? shareAccessKey() : null));
  const [keyDraft, setKeyDraft] = useState<string | null>(null);
  const [ttl, setTtl] = useState<ShareTtl>(SHARE_TTLS[0]);
  const upload = useRef<AbortController | null>(null);

  useEffect(() => {
    setLink(savedShare(key));
    setError(null);
    return () => upload.current?.abort();
  }, [key]);

  // The link disappears from here when it expires.
  useEffect(() => {
    if (!link || link.expiresAt === null) return;
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
      const access = accessKey ? { key: accessKey, ttl } : null;
      const made = await uploadShare(host.shareOrigin, file, basename(path), shareView(frame), access, setProgress, ctl.signal);
      saveShare(key, made);
      setLink(made);
    } catch (e) {
      // Wrong key: forget it on this device so the field comes back.
      if (e instanceof ShareError && e.code === "forbidden") rememberKey(null);
      if (!ctl.signal.aborted) setError(describe(e));
    } finally {
      if (upload.current === ctl) upload.current = null;
      setProgress(null);
    }
  }

  function rememberKey(value: string | null) {
    setShareAccessKey(value);
    setAccessKey(value);
    setKeyDraft(null);
    setError(null);
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
      /* no permission: the field is selectable */
    }
  }

  const expires = link?.expiresAt != null ? new Date(link.expiresAt) : null;
  const now = new Date();
  const sameDay = expires?.toDateString() === now.toDateString();
  const when = expires?.toLocaleString(locale, {
    ...(sameDay ? {} : { day: "2-digit", month: "2-digit" }),
    ...(expires && expires.getFullYear() !== now.getFullYear() ? { year: "numeric" } : {}),
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
              <p className="help">{expires ? ts.expires(when ?? "") : ts.never}</p>
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
          {accessKey ? (
            <div className="share-row">
              <span className="help">{ts.ttlLabel}</span>
              <div className="share-ttl">
                <Select<ShareTtl>
                  value={ttl}
                  options={SHARE_TTLS.map((v) => ({ value: v, label: ts.ttl(v) }))}
                  onChange={setTtl}
                  aria-label={ts.ttlLabel}
                />
              </div>
            </div>
          ) : keyDraft !== null ? (
            <form
              className="share-row"
              onSubmit={(e) => {
                e.preventDefault();
                if (keyDraft.trim()) rememberKey(keyDraft.trim());
              }}
            >
              <input
                className="share-url"
                type="password"
                autoComplete="off"
                autoFocus
                placeholder={ts.keyPlaceholder}
                aria-label={ts.keyPlaceholder}
                value={keyDraft}
                onChange={(e) => setKeyDraft(e.currentTarget.value)}
              />
              <button className="btn-sm" type="submit" disabled={!keyDraft.trim()}>
                {ts.keySave}
              </button>
              <button className="btn-sm ghost" type="button" onClick={() => setKeyDraft(null)}>
                {t.common.cancel}
              </button>
            </form>
          ) : null}
          <div className="trim-actions">
            <button className="btn-sm" onClick={create}>
              {ts.create}
            </button>
            <button className="btn-sm ghost" onClick={onClose}>
              {t.common.cancel}
            </button>
          </div>
          {accessKey ? (
            <button className="link share-key-toggle" onClick={() => rememberKey(null)} title={ts.forgetKeyTitle}>
              {ts.forgetKey}
            </button>
          ) : (
            shareAdmin &&
            keyDraft === null && (
              <button className="link share-key-toggle" onClick={() => setKeyDraft("")}>
                {ts.haveKey}
              </button>
            )
          )}
        </>
      )}
      {error && <p className="help export-error">{error}</p>}
    </div>
  );
}

/** QR code of the link, to open on a phone. The library is loaded only here. */
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
  // 2-module margin and white background: QR readers need the contrast.
  return (
    <svg className="share-qr" viewBox={`-2 -2 ${qr.n + 4} ${qr.n + 4}`} shapeRendering="crispEdges" role="img">
      <rect x={-2} y={-2} width={qr.n + 4} height={qr.n + 4} fill="#fff" />
      <path d={qr.d} fill="#000" />
    </svg>
  );
}
