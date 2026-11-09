import React, { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

const TREATMENTS = [
  ["drop_duplicates", "Remove duplicate rows", "Deletes rows that are exact copies of another row."],
  ["fill_missing", "Fill missing values", "Numbers get the median, text gets the most common value."],
  ["drop_empty_rows", "Remove rows with missing values", "Deletes any row that has a blank cell."],
  ["clean_text", "Trim text", "Removes stray spaces around text."],
  ["clip_outliers", "Cap extreme values", "Pulls very unusual numbers back to a sensible limit (IQR rule)."],
  ["normalise", "Rescale numbers to 0..1", "Puts all numeric columns on the same scale."],
];

const fmtBytes = (n) => {
  if (!n) return "0 B";
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), 3);
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${["B", "KB", "MB", "GB"][i]}`;
};

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}
const postJson = (path, data) =>
  api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

function AddData({ onAdded }) {
  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const run = async (job) => {
    setBusy(true);
    setError("");
    try {
      const { id } = await job();
      setUrl("");
      setName("");
      onAdded(id);
    } catch (e) {
      setError(e.message);
    }
    setBusy(false);
  };
  const addLink = (e) => {
    e.preventDefault();
    run(() => postJson("/api/datasets/link", { url, name }));
  };
  const addFile = (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    form.append("name", name);
    run(() => api("/api/datasets/upload", { method: "POST", body: form }));
  };

  return (
    <section className="card">
      <h2>Add a dataset</h2>
      <form className="add" onSubmit={addLink}>
        <label>Dataset link
          <input type="url" required placeholder="https://example.org/data.csv" value={url}
            onChange={(e) => setUrl(e.target.value)} />
        </label>
        <label>Name (optional)
          <input value={name} onChange={(e) => setName(e.target.value)} />
        </label>
        <button disabled={busy}>Download</button>
        <label className="button ghost">Upload a file
          <input type="file" hidden disabled={busy} onChange={addFile} />
        </label>
      </form>
      <p className="hint">Accepts CSV, TSV, JSON, Parquet, images, or a ZIP, which is unpacked for you. Limit 5 GB per link.</p>
      {error && <p className="error">{error}</p>}
    </section>
  );
}

function Profile({ data }) {
  return (
    <>
      <div className="stats">
        {[["Rows", data.rows], ["Columns", data.cols], ["Duplicate rows", data.duplicates],
          ["Missing cells", data.missing_cells]].map(([k, v]) => (
          <div key={k}><b>{v.toLocaleString()}</b><span>{k}</span></div>
        ))}
      </div>
      <h3>Columns</h3>
      <div className="scroll"><table>
        <thead><tr><th>Name</th><th>Type</th><th>Missing</th><th>Unique</th><th>Min</th><th>Mean</th><th>Max</th></tr></thead>
        <tbody>{data.columns.map((c) => (
          <tr key={c.name}><td>{c.name}</td><td>{c.type}</td><td>{c.missing.toLocaleString()}</td>
            <td>{c.unique.toLocaleString()}</td><td>{c.min ?? ""}</td><td>{c.mean ?? ""}</td><td>{c.max ?? ""}</td></tr>
        ))}</tbody>
      </table></div>
      <h3>First rows</h3>
      <div className="scroll"><table>
        <thead><tr>{data.columns.map((c) => <th key={c.name}>{c.name}</th>)}</tr></thead>
        <tbody>{data.preview.map((row, i) => (
          <tr key={i}>{data.columns.map((c) => <td key={c.name}>{String(row[c.name] ?? "")}</td>)}</tr>
        ))}</tbody>
      </table></div>
    </>
  );
}

function Detail({ id, onChanged }) {
  const [ds, setDs] = useState(null);
  const [file, setFile] = useState("");
  const [prof, setProf] = useState(null);
  const [ops, setOps] = useState([]);
  const [log, setLog] = useState([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    const d = await api(`/api/datasets/${id}`);
    setDs(d);
    return d;
  }, [id]);

  useEffect(() => {
    setFile(""); setProf(null); setLog([]); setError("");
    load();
  }, [id, load]);

  useEffect(() => {
    if (!ds || !["downloading", "unpacking"].includes(ds.status)) return;
    const t = setInterval(() => load().then(onChanged), 1500);
    return () => clearInterval(t);
  }, [ds, load, onChanged]);

  useEffect(() => {
    if (!file && ds) {
      const first = ds.files.find((f) => f.tabular);
      if (first) setFile(first.path);
    }
  }, [ds, file]);

  useEffect(() => {
    if (!file) return;
    setBusy("Reading the file"); setProf(null); setError("");
    api(`/api/datasets/${id}/profile?file=${encodeURIComponent(file)}`)
      .then(setProf).catch((e) => setError(e.message)).finally(() => setBusy(""));
  }, [id, file]);

  const apply = async () => {
    setBusy("Treating the data"); setError("");
    try {
      const r = await postJson(`/api/datasets/${id}/treat`, { file, ops: ops.map((op) => ({ op })) });
      setLog(r.log); setOps([]);
      await load();
      setFile(r.file);
      onChanged();
    } catch (e) { setError(e.message); }
    setBusy("");
  };

  const remove = async () => {
    if (!window.confirm(`Delete "${ds.name}" and all its files?`)) return;
    await api(`/api/datasets/${id}`, { method: "DELETE" });
    onChanged(null);
  };

  if (!ds) return <section className="card">Loading</section>;
  const dl = (f) => `/api/datasets/${id}/download${f ? `?file=${encodeURIComponent(f)}` : ""}`;

  return (
    <section className="card">
      <div className="head">
        <div><h2>{ds.name}</h2><p className="hint">{ds.source}</p></div>
        {ds.status === "ready" && <a className="button ghost" href={dl("")}>{ds.builtin ? "Download CSV" : "Download all"}</a>}
        {!ds.builtin && <button className="ghost danger" onClick={remove}>Delete</button>}
      </div>
      {ds.status === "failed" && <p className="error">{ds.error}</p>}
      {["downloading", "unpacking"].includes(ds.status) &&
        <p>{ds.status === "downloading" ? `Downloading: ${fmtBytes(ds.bytes)} so far` : "Unpacking"}</p>}
      {ds.status === "ready" && <>
        <h3>Files ({ds.n_files.toLocaleString()})</h3>
        <ul className="files">{ds.files.slice(0, 12).map((f) => (
          <li key={f.path}>
            {f.tabular ? <button className={`link ${f.path === file ? "on" : ""}`} onClick={() => setFile(f.path)}>{f.path}</button> : <span>{f.path}</span>}
            <em>{fmtBytes(f.bytes)}</em><a href={dl(f.path)}>Download</a>
          </li>
        ))}</ul>
        {ds.n_files > 12 && <p className="hint">Showing 12 of {ds.n_files.toLocaleString()} files. Use Download all to get every file.</p>}
        {!ds.files.some((f) => f.tabular) && <p className="hint">No table found, so there is nothing to treat here.</p>}
        {file && <>
          <h3>Treat {file}</h3>
          <div className="ops">{TREATMENTS.map(([key, label, help]) => (
            <label key={key}>
              <input type="checkbox" checked={ops.includes(key)}
                onChange={(e) => setOps(e.target.checked ? [...ops, key] : ops.filter((o) => o !== key))} />
              <span><b>{label}</b>{help}</span>
            </label>
          ))}</div>
          <button disabled={!ops.length || !!busy} onClick={apply}>Apply and save a new copy</button>
          <p className="hint">Your original file is never changed.</p>
          {log.length > 0 && <ul className="log">{log.map((l) => <li key={l}>{l}</li>)}</ul>}
        </>}
        {busy && <p>{busy}...</p>}
        {error && <p className="error">{error}</p>}
        {prof && <Profile data={prof} />}
      </>}
    </section>
  );
}

function DataTools() {
  const [list, setList] = useState([]);
  const [selected, setSelected] = useState("nih-processed");
  const refresh = useCallback(() => api("/api/datasets").then(setList), []);
  useEffect(() => { refresh(); }, [refresh]);
  const changed = useCallback((next) => {
    if (next === null) setSelected("nih-processed");
    refresh();
  }, [refresh]);

  return (
    <div className="tools">
      <AddData onAdded={(id) => { setSelected(id); refresh(); }} />
      <div className="split">
        <section className="card list">
          <h2>Datasets</h2>
          {list.map((d) => (
            <button key={d.id} className={`row ${d.id === selected ? "on" : ""}`} onClick={() => setSelected(d.id)}>
              <b>{d.name}</b><span>{d.status === "ready" ? `${d.n_files} files` : d.status}</span>
            </button>
          ))}
        </section>
        <Detail id={selected} onChanged={changed} />
      </div>
    </div>
  );
}

const REPORT = [
  ["Full report", ""], ["Task", "task"], ["Data and sources", "data"], ["What the data is", "what"],
  ["Examples", "examples"], ["Visualisations", "viz"], ["Pipeline", "pipeline"],
  ["Model plan", "model"], ["Risks", "risks"],
];

function App() {
  const [view, setView] = useState("tools");
  return (
    <div className="shell">
      <nav>
        <h1>Chest X-ray project</h1>
        <p>Data tools</p>
        <button className={view === "tools" ? "on" : ""} onClick={() => setView("tools")}>Add, treat, download</button>
        <p>Report</p>
        {REPORT.map(([label, anchor]) => (
          <button key={label} className={view === `r:${anchor}` ? "on" : ""} onClick={() => setView(`r:${anchor}`)}>{label}</button>
        ))}
      </nav>
      <main>
        {view === "tools" ? <DataTools /> :
          <iframe title="Project report" key={view} src={`/report#${view.slice(2)}`} />}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
