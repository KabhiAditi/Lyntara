import './style.css';

/* ── API CONFIG ── */
const API_BASE = 'http://localhost:8000/api';
let sessionId = null;
let connectedDbName = '';

function getDbCredentials() {
  return {
    engine: selectedEngine,
    host: document.getElementById('inp-host').value,
    port: Number(document.getElementById('inp-port').value || 0) || null,
    database: document.getElementById('inp-dbname').value,
    username: document.getElementById('inp-user').value,
    password: document.getElementById('inp-pass').value,
  };
}

// selectedEngine is declared later but hoisted via var
var selectedEngine;

function escHtml(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function renderSQL(sql) {
  if (!sql) return '<span class="cmt">-- No SQL generated</span>';
  let x = escHtml(sql);
  const kw = [
    'SELECT','FROM','WHERE','JOIN','LEFT','RIGHT','INNER','OUTER','ON',
    'GROUP BY','ORDER BY','LIMIT','HAVING','WITH','AS','AND','OR','DISTINCT'
  ];
  kw.forEach(k => {
    const re = new RegExp('\\b' + k.replace(' ', '\\s+') + '\\b', 'gi');
    x = x.replace(re, (m) => '<span class="kw">' + m.toUpperCase() + '</span>');
  });
  return x;
}

async function apiPost(path, body) {
  const res = await fetch(API_BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {})
  });
  const txt = await res.text();
  let json = null;
  try { json = txt ? JSON.parse(txt) : null; } catch (_) {}
  if (!res.ok) {
    const msg = (json && (json.message || json.detail || json.error)) ? JSON.stringify(json) : (txt || ('HTTP ' + res.status));
    throw new Error(msg);
  }
  return json;
}

async function apiGet(path) {
  const res = await fetch(API_BASE + path, { method: 'GET' });
  const txt = await res.text();
  let json = null;
  try { json = txt ? JSON.parse(txt) : null; } catch (_) {}
  if (!res.ok) {
    const msg = (json && (json.message || json.detail || json.error)) ? JSON.stringify(json) : (txt || ('HTTP ' + res.status));
    throw new Error(msg);
  }
  return json;
}

function renderSchemaExplorer(schemaData) {
  const grid = document.getElementById('schema-grid');
  if (!grid) return;
  const tables = Object.keys(schemaData || {});
  if (tables.length === 0) {
    grid.innerHTML = '<div class="schema-card"><div class="schema-card-head"><div class="schema-tname">No schema</div><div class="schema-cnt">0 columns</div></div><div class="schema-cols-list"><div class="schema-col-row"><span class="schema-col-name">No data available</span><span class="schema-col-type">N/A</span></div></div></div>';
    return;
  }

  grid.innerHTML = tables.map(function(tableName){
    const info = schemaData[tableName] || {};
    const columns = Array.isArray(info.columns) ? info.columns : [];
    const pks = new Set(Array.isArray(info.primary_keys) ? info.primary_keys : []);
    const fkCols = new Set(Array.isArray(info.foreign_keys) ? info.foreign_keys.map(f => f.column) : []);
    const colsHtml = columns.map(function(c){
      const colName = c && c.column_name ? String(c.column_name) : '';
      const dataType = c && c.data_type ? String(c.data_type).toUpperCase() : 'UNKNOWN';
      const tag = pks.has(colName)
        ? '<span class="pk-tag">PK</span>'
        : (fkCols.has(colName) ? '<span class="fk-tag">FK</span>' : '');
      return '<div class="schema-col-row"><span class="schema-col-name">' + escHtml(colName) + '</span><div style="display:flex;gap:4px;align-items:center">' + tag + '<span class="schema-col-type">' + escHtml(dataType) + '</span></div></div>';
    }).join('');

    return '<div class="schema-card"><div class="schema-card-head"><div class="schema-tname">' + escHtml(tableName) + '</div><div class="schema-cnt">' + columns.length + ' columns</div></div><div class="schema-cols-list">' + colsHtml + '</div></div>';
  }).join('');
}

/* ── DB TYPE SELECT ── */
selectedEngine = 'postgresql';
window.selectDB = function(el, engine, port) {
  document.querySelectorAll('.db-btn').forEach(b => {
    b.classList.remove('active');
    b.querySelectorAll('circle,path,rect,ellipse').forEach(s => s.setAttribute('stroke','#8b909e'));
  });
  el.classList.add('active');
  el.querySelectorAll('circle,path,rect,ellipse').forEach(s => s.setAttribute('stroke','#19c490'));
  selectedEngine = engine;
  document.getElementById('inp-port').value = port;
  if (engine === 'sqlite') {
    document.getElementById('inp-host').value = 'local file';
    document.getElementById('inp-dbname').value = 'database.db';
    document.getElementById('inp-user').value = '';
    document.getElementById('inp-pass').value = '';
  } else {
    document.getElementById('inp-host').value = 'localhost';
    document.getElementById('inp-dbname').value = 'ecommerce_db';
    document.getElementById('inp-user').value = 'root';
  }
};

/* ── TEST CONNECTION ── */
window.testConn = async function() {
  var s = document.getElementById('conn-status');
  var m = document.getElementById('conn-msg');
  s.classList.remove('err');
  m.textContent = 'Testing connection…';
  s.classList.add('show');

  try {
    const out = await apiPost('/connect', getDbCredentials());
    if (out.ok) {
      s.classList.remove('err');
      m.textContent = 'Connected (Qdrant + PostgreSQL validated)';
    } else {
      s.classList.add('err');
      m.textContent = (out.message || 'Connection failed');
    }
  } catch (e) {
    s.classList.add('err');
    m.textContent = 'Connection failed — ' + (e && e.message ? e.message : 'unknown error');
  }
};

/* ── LOADING SCREEN HELPERS ── */
var checkSVG = '<svg class="pp-icon" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6" stroke="#19c490" stroke-width="1.5"/><path d="M5 8l2 2 4-4" stroke="#19c490" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';
var spinHTML = '<div class="spinner"></div>';
var greyCircle = '<svg class="pp-icon" viewBox="0 0 16 16" fill="none"><circle cx="8" cy="8" r="6" stroke="#555b6a" stroke-width="1.5"/></svg>';

function resetPipelineUI() {
  ['pp1','pp2','pp3','pp4','pp5'].forEach(function(id, i) {
    var el = document.getElementById(id);
    el.classList.remove('done','doing');
    var labels = ['Connecting to database','Extracting schema','Generating context','Upserting embeddings','Pipeline ready'];
    el.innerHTML = (i === 0 ? spinHTML : greyCircle) + '<span>' + labels[i] + '</span>';
  });
}

function markStep(id, state, label) {
  var el = document.getElementById(id);
  el.classList.remove('done','doing');
  if (state === 'done') {
    el.classList.add('done');
    el.innerHTML = checkSVG + '<span>' + label + '</span>';
  } else if (state === 'doing') {
    el.classList.add('doing');
    el.innerHTML = spinHTML + '<span>' + label + '</span>';
  }
}

async function enterApp(dbname) {
  connectedDbName = dbname;
  var eng = selectedEngine.charAt(0).toUpperCase() + selectedEngine.slice(1);
  document.getElementById('topbar-db').textContent = dbname + ' · ' + eng;
  try {
    var schemaPayload = await apiGet('/schema?database=' + encodeURIComponent(dbname));
    if (schemaPayload && schemaPayload.ok) {
      renderSchemaExplorer(schemaPayload.schema_data || {});
      document.getElementById('schema-meta').textContent = dbname + ' · ' + eng + ' · ' + String(schemaPayload.table_count || 0) + ' tables';
    } else {
      document.getElementById('schema-meta').textContent = dbname + ' · ' + eng + ' · schema unavailable';
    }
  } catch (e) {
    document.getElementById('schema-meta').textContent = dbname + ' · ' + eng + ' · schema unavailable';
  }
  show('s-app');
}

/* ── GO TO LOADING SCREEN ── */
window.goLoading = async function() {
  var dbname = document.getElementById('inp-dbname').value || 'database';
  var creds = getDbCredentials();

  // 1. Check if pipeline already done (redundancy)
  document.getElementById('loading-sub').textContent = 'Checking index for ' + dbname + '…';
  show('s-loading');
  resetPipelineUI();

  try {
    var statusRes = await apiPost('/pipeline/status', creds);
    if (statusRes.already_indexed) {
      // Skip pipeline — already indexed
      ['pp1','pp2','pp3','pp4','pp5'].forEach(function(id) {
        var el = document.getElementById(id);
        el.classList.add('done');
        el.innerHTML = checkSVG + '<span>' + el.querySelector('span').textContent + '</span>';
      });
      document.getElementById('loading-sub').textContent = 'Already indexed — loading app…';
      setTimeout(function() { enterApp(dbname); }, 600);
      return;
    }
  } catch (e) {
    // Status check failed — proceed with pipeline anyway
  }

  // 2. Run the real pipeline via SSE stream
  document.getElementById('loading-sub').textContent = 'Connecting to ' + dbname + '…';
  markStep('pp1', 'doing', 'Connecting to database');

  try {
    var response = await fetch(API_BASE + '/pipeline/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(creds),
    });

    if (!response.ok) {
      throw new Error('Pipeline request failed: HTTP ' + response.status);
    }

    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = '';

    while (true) {
      var result = await reader.read();
      if (result.done) break;
      buffer += decoder.decode(result.value, { stream: true });

      // Parse SSE events from buffer
      var lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line
      for (var li = 0; li < lines.length; li++) {
        var line = lines[li];
        if (line.startsWith('data: ')) {
          try {
            var evt = JSON.parse(line.substring(6));
            document.getElementById('loading-sub').textContent = evt.message || '';

            if (evt.step === 'error') {
              show('s-connect');
              var cs = document.getElementById('conn-status');
              var cm = document.getElementById('conn-msg');
              cs.classList.add('show', 'err');
              cm.textContent = evt.message;
              return;
            }

            // Map pipeline steps to UI
            if (evt.step === 'extracting_schema') {
              if (evt.progress >= 20) {
                markStep('pp1', 'done', 'Connected to database');
                markStep('pp2', 'doing', 'Extracting schema');
              }
            } else if (evt.step === 'generating_context') {
              markStep('pp2', 'done', 'Schema extracted');
              markStep('pp3', 'doing', evt.message || 'Generating context…');
            } else if (evt.step === 'upserting_embeddings') {
              markStep('pp3', 'done', 'Context generated');
              markStep('pp4', 'doing', 'Upserting embeddings…');
            } else if (evt.step === 'creating_indexes') {
              markStep('pp4', 'done', 'Embeddings upserted');
              markStep('pp5', 'doing', 'Creating indexes…');
            } else if (evt.step === 'done') {
              markStep('pp5', 'done', 'Pipeline ready');
            }
          } catch(_) {}
        }
      }
    }

    // Pipeline complete, enter app
    setTimeout(function() { enterApp(dbname); }, 500);

  } catch (e) {
    show('s-connect');
    var s2 = document.getElementById('conn-status');
    var m2 = document.getElementById('conn-msg');
    s2.classList.add('show', 'err');
    m2.textContent = 'Pipeline error — ' + (e.message || 'unknown');
  }
};

/* ── SCREEN SWITCH ── */
function show(id) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
}
window.disconnect = function() { show('s-connect'); };

/* ── TAB SWITCH ── */
window.switchTab = function(tab, btn) {
  document.querySelectorAll('.nav-pill').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab-' + tab).classList.add('active');
};

/* ── HISTORY DATA ── */
var historyData = [
  {
    query: "Welcome to QueryNL2SQL!",
    type: "SRD", note: "Ready to query the database",
    tables: [{name:"instructions", cols:"type a query below"}],
    sql: '<span class="kw">SELECT</span> <span class="val">*</span> <span class="kw">FROM</span> users <span class="kw">LIMIT</span> <span class="val">1</span>;',
    raw_sql: 'SELECT * FROM users LIMIT 1;',
    thead: ["status"],
    tbody: [["Type a natural language query below and hit Run query."]]
  }
];

let activeIdx = 0;

window.loadHistory = function(el, idx) {
  if (!historyData[idx]) return;
  activeIdx = idx;
  document.querySelectorAll('.hist-li').forEach(h => h.classList.remove('active'));
  if (el) el.classList.add('active');
  var d = historyData[idx];
  document.getElementById('q-text').textContent = d.query;
  var typeChip = document.getElementById('q-type-chip');
  typeChip.textContent = d.type === 'SRD' ? 'SRD — single-round query' : 'MRD — multi-round query';
  typeChip.className = 'chip ' + (d.type ? d.type.toLowerCase() : 'srd');
  document.getElementById('q-note').textContent = d.note;
  var tablesHTML = d.tables.map(function(t){
    return '<div class="tbl-chip"><div class="tbl-name">'+t.name+'</div><div class="tbl-cols">'+t.cols+'</div></div>';
  }).join('');
  document.getElementById('tables-matched').innerHTML = tablesHTML;
  document.getElementById('sql-display').innerHTML = d.sql;
  var thead = d.thead.map(function(h){ return '<th>'+h+'</th>'; }).join('');
  var tbody = d.tbody.map(function(r){
    return '<tr>'+r.map(function(c){ return '<td>'+c+'</td>'; }).join('')+'</tr>';
  }).join('');
  document.querySelector('.res-table thead tr').innerHTML = thead;
  document.getElementById('res-body').innerHTML = tbody;
};

function renderHistorySidebar() {
  const el = document.getElementById('hist-list');
  el.innerHTML = historyData.map(function(d, i){
    const tagClass = d.type === 'MRD' ? 'mrd' : (d.type==='ERROR' ? 'mrd':'srd'); 
    return '<div class="hist-li' + (i === activeIdx ? ' active' : '') + '" onclick="loadHistory(this, ' + i + ')">' +
      '<span class="htag ' + tagClass + '">' + escHtml((d.type||'').substring(0,3)) + '</span>' + 
      escHtml(d.query.substring(0,32) + (d.query.length > 32 ? '…' : '')) + '</div>';
  }).join('');
}

/* ── RUN QUERY ── */
window.runQuery = async function() {
  var input = document.getElementById('nl-input');
  var q = input.value.trim();
  if (!q) return;

  var newEntry = {
    query: q,
    type: "SRD", note: "Generating SQL...",
    tables: [{name: "...", cols: "matching tables via vector search"}],
    sql: '<span class="cmt">-- Generating SQL for: ' + escHtml(q) + '</span>',
    raw_sql: '',
    thead: ["status", "message"],
    tbody: [["executing", "SQL is being generated."]]
  };

  historyData.unshift(newEntry);
  activeIdx = 0;
  renderHistorySidebar();
  window.loadHistory(document.querySelector('.hist-li.active'), 0);

  try {
    const out = await apiPost('/query', { query: q, top_k: 5, session_id: sessionId, database: connectedDbName });
    sessionId = out.session_id;

    const type = out.query_type || 'UNKNOWN';
    const warnings = Array.isArray(out.warnings) ? out.warnings : [];
    const baseNote = (type === 'MRD' ? 'Multi-round · combined with previous turns' : 'Standalone · no prior context needed');
    
    newEntry.type = type;
    newEntry.resolved_query = out.resolved_query || q;
    newEntry.note = warnings.length ? (baseNote + ' · ' + warnings[0]) : baseNote;
    
    const schema = out.schema || {};
    const tbls = Object.keys(schema);
    if (tbls.length === 0) {
      newEntry.tables = [{name: "No match", cols: "No schema context found"}];
    } else {
      newEntry.tables = tbls.map(function(t){
        const cols = (schema[t] && schema[t].columns) ? schema[t].columns.map(c => c.column_name).slice(0,10).join(', ') : '';
        return {name: t, cols: cols || '...' };
      });
    }

    if (out.sql) {
      newEntry.sql = renderSQL(out.sql);
      newEntry.raw_sql = out.sql; 
    } else {
      const firstWarn = (warnings.length > 0) ? warnings[0] : 'No SQL generated (schema mismatch or model unavailable)';
      newEntry.sql = '<span class="cmt">-- ' + escHtml(firstWarn) + '</span>';
      newEntry.raw_sql = '';
    }

    newEntry.thead = ["status", "message", "session_id", "sql_valid"];
    newEntry.tbody = [
      ["ready", warnings.length ? warnings.join(' | ') : "SQL is ready. Click Execute to run.", out.session_id, String(!!out.sql_valid)]
    ];

  } catch (e) {
    newEntry.type = "ERROR";
    newEntry.note = "Backend error (see results panel)";
    newEntry.tables = [{name:"error", cols:"failed to fetch schema/sql"}];
    newEntry.sql = '<span class="cmt">-- Backend error: ' + escHtml(e && e.message ? e.message : 'unknown') + '</span>';
    newEntry.raw_sql = '';
    newEntry.thead = ['error'];
    newEntry.tbody = [['Backend error — ' + escHtml(e && e.message ? e.message : 'unknown')]];
  }

  renderHistorySidebar();
  window.loadHistory(document.querySelector('.hist-li.active'), 0);
  input.value = '';
};

/* ── EXECUTE SQL ── */
window.executeSQL = async function() {
  const d = historyData[activeIdx];
  if (!d || !d.raw_sql) {
      alert("No valid SQL found to execute!");
      return;
  }
  
  const payload = {
    engine: selectedEngine,
    host: document.getElementById('inp-host').value,
    port: Number(document.getElementById('inp-port').value || 0) || null,
    database: document.getElementById('inp-dbname').value,
    username: document.getElementById('inp-user').value,
    password: document.getElementById('inp-pass').value,
    sql: d.raw_sql,
    nl_query: d.resolved_query || d.query
  };

  document.querySelector('.res-table thead tr').innerHTML = '<th>status</th>';
  document.getElementById('res-body').innerHTML = '<tr><td style="color:var(--teal)">Executing query on db...</td></tr>';

  try {
    const out = await apiPost('/execute', payload);
    if (!out.ok) {
       d.thead = ["error"];
       if (out.fixed_sql) {
         d.tbody = [[out.error + " | Gemini attempted a fix! Review the new query above and click Execute again."]];
         d.sql = renderSQL(out.fixed_sql);
         d.raw_sql = out.fixed_sql;
       } else {
         d.tbody = [[out.error]];
       }
    } else if (out.columns && out.columns.length > 0) {
       d.thead = out.columns;
       d.tbody = out.rows.length ? out.rows : [["No rows returned."]];
    } else {
       d.thead = ["status"];
       d.tbody = [["Query executed successfully (no rows returned)."]];
    }
  } catch (e) {
       d.thead = ["error"];
       d.tbody = [[e.message]];
  }
  window.loadHistory(document.querySelector('.hist-li.active'), activeIdx);
};

window.handleKey = function(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); window.runQuery(); }
};

/* ── COPY SQL ── */
window.copySQL = function() {
  var text = document.getElementById('sql-display').innerText;
  navigator.clipboard.writeText(text).catch(function(){});
  var btn = document.querySelector('.copy-btn');
  btn.textContent = 'copied!';
  setTimeout(function(){ btn.textContent = 'copy'; }, 1500);
};

/* ── RESET INDEX (demo replay) ── */
window.resetIndex = async function() {
  if (!connectedDbName) {
    alert('No database connected.');
    return;
  }
  if (!confirm('Delete Qdrant collections for "' + connectedDbName + '"? You can re-index by connecting again.')) return;
  try {
    await apiPost('/pipeline/delete', { database: connectedDbName });
    connectedDbName = '';
    sessionId = null;
    show('s-connect');
  } catch (e) {
    alert('Reset failed: ' + (e.message || 'unknown error'));
  }
};
