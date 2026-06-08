'use strict';

// ── Seed users ─────────────────────────────────────────────────────────────────
const SEED_USERS = [
  { id:1, name:'Quản trị viên',   email:'admin@luatmatuy.vn',  password:'admin123', role:'admin',  joined:'2024-01-10', status:'active' },
  { id:2, name:'Nguyễn Văn Minh', email:'nguyen@phapluat.vn',  password:'demo123',  role:'user',   joined:'2024-03-18', status:'active' },
  { id:3, name:'Trần Thị Hoa',    email:'tran@tuvanluat.vn',   password:'demo123',  role:'user',   joined:'2024-05-22', status:'active' },
  { id:4, name:'Lê Quang Khải',   email:'le@nhanquyen.vn',     password:'demo123',  role:'viewer', joined:'2024-07-01', status:'active' },
];

function loadUsers() {
  try { const s = localStorage.getItem('luatmatuy-users'); if (s) return JSON.parse(s); } catch {}
  localStorage.setItem('luatmatuy-users', JSON.stringify(SEED_USERS));
  return [...SEED_USERS];
}
function saveUsers(u) { localStorage.setItem('luatmatuy-users', JSON.stringify(u)); }
function getInitials(n) { return (n||'?').split(' ').map(w=>w[0]).join('').slice(0,2).toUpperCase(); }
function avatarClass(r) { return r==='admin'?'role-admin':r==='viewer'?'role-viewer':''; }

// ── State ─────────────────────────────────────────────────────────────────────
let state = {
  theme:'light', currentUser:null, userEditId:null,
  messages:[], isTyping:false, chatStarted:false,
  sessions:[
    {id:1, title:'Hình phạt tội vận chuyển ma tuý'},
    {id:2, title:'Quy trình cai nghiện bắt buộc'},
    {id:3, title:'Xử phạt hành chính sử dụng ma tuý'},
    {id:4, title:'Phân loại chất ma tuý Danh mục I'},
    {id:5, title:'Quyền người sau cai nghiện'},
  ],
  activeSession:null,
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const chatArea         = $('chatArea');
const welcomeScr       = $('welcomeScreen');
const msgContainer     = $('messagesContainer');
const textarea         = $('msgInput');
const sendBtn          = $('sendBtn');
const sidebar          = $('sidebar');
const overlay          = $('overlay');
const settingsPanel    = $('settingsPanel');
const settingsBackdrop = $('settingsBackdrop');
const userMgmtPanel    = $('userMgmtPanel');
const userMgmtBackdrop = $('userMgmtBackdrop');
const userDropdown     = $('userDropdown');

// ── Init ──────────────────────────────────────────────────────────────────────
function init() {
  const savedTheme = localStorage.getItem('luatmatuy-theme') || 'light';
  state.theme = savedTheme;
  applyTheme(savedTheme, true);
  buildHistory();
  attachEvents();
  autoResize(textarea);
  if (checkSession()) { hideAuthScreen(); updateTopbarUser(); }
}

// ── Theme ─────────────────────────────────────────────────────────────────────
function applyTheme(t, instant=false) {
  if (!instant) { document.documentElement.classList.add('theme-transitioning'); setTimeout(()=>document.documentElement.classList.remove('theme-transitioning'), 260); }
  document.documentElement.setAttribute('data-theme', t==='system' ? (matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light') : t);
  state.theme = t;
  localStorage.setItem('luatmatuy-theme', t);
  document.querySelectorAll('.theme-opt').forEach(b => b.classList.toggle('active', b.dataset.themeVal===t));
}

// ── Auth ──────────────────────────────────────────────────────────────────────
function checkSession() {
  try { const s=JSON.parse(localStorage.getItem('luatmatuy-session')); if(!s) return false; const u=loadUsers().find(x=>x.id===s.id); if(!u||u.status==='inactive') return false; state.currentUser=u; return true; } catch{return false;}
}
function loginUser(email,password) {
  const u=loadUsers().find(u=>u.email.toLowerCase()===email.toLowerCase()&&u.password===password);
  if(!u) throw new Error('Email hoặc mật khẩu không đúng.');
  if(u.status==='inactive') throw new Error('Tài khoản này đã bị vô hiệu hóa.');
  state.currentUser=u; localStorage.setItem('luatmatuy-session',JSON.stringify(u)); return u;
}
function registerUser(name,email,password) {
  const users=loadUsers();
  if(users.find(u=>u.email.toLowerCase()===email.toLowerCase())) throw new Error('Email này đã được đăng ký.');
  const nu={id:Date.now(),name,email,password,role:'user',joined:new Date().toISOString().slice(0,10),status:'active'};
  users.push(nu); saveUsers(users); state.currentUser=nu;
  localStorage.setItem('luatmatuy-session',JSON.stringify(nu)); return nu;
}
function logoutUser() {
  state.currentUser=null; localStorage.removeItem('luatmatuy-session');
  state.chatStarted=false; state.messages=[];
  msgContainer.innerHTML='';
  welcomeScr.style.display=''; msgContainer.style.display='none';
  buildHistory(); showAuthScreen();
}
function showAuthScreen() { const s=$('authScreen'); if(!s){location.reload();return;} s.style.display=''; s.classList.remove('hiding'); void s.offsetWidth; }
function hideAuthScreen() { const s=$('authScreen'); if(!s)return; s.classList.add('hiding'); setTimeout(()=>{s.style.display='none';},260); }

function updateTopbarUser() {
  const u=state.currentUser; if(!u)return;
  const init=getInitials(u.name); const ac=avatarClass(u.role);
  const av=$('topbarAvatar'); const nm=$('topbarName');
  if(av){av.textContent=init;av.className=`topbar-user-avatar ${ac}`;}
  if(nm) nm.textContent=u.name.split(' ').slice(-1)[0];
  const dav=$('dropdownAvatar'); const dnm=$('dropdownName'); const dem=$('dropdownEmail'); const drl=$('dropdownRoleBadge');
  if(dav){dav.textContent=init;dav.className=`dropdown-avatar ${ac}`;}
  if(dnm) dnm.textContent=u.name;
  if(dem) dem.textContent=u.email;
  if(drl){drl.textContent=u.role==='admin'?'Quản trị viên':u.role==='viewer'?'Xem':'Người dùng';drl.className=`role-badge role-${u.role}`;}
  document.querySelectorAll('.admin-only').forEach(el=>el.style.display=u.role==='admin'?'':'none');
}

// ── Events ────────────────────────────────────────────────────────────────────
function attachEvents() {
  // Auth tabs
  $('tabLogin').addEventListener('click',()=>{ $('tabLogin').classList.add('active'); $('tabRegister').classList.remove('active'); $('loginSection').style.display=''; $('registerSection').style.display='none'; });
  $('tabRegister').addEventListener('click',()=>{ $('tabRegister').classList.add('active'); $('tabLogin').classList.remove('active'); $('registerSection').style.display=''; $('loginSection').style.display='none'; });

  // Login
  $('loginSubmit').addEventListener('click',()=>{
    const email=$('loginEmail').value.trim(); const pass=$('loginPassword').value;
    const err=$('loginError'); err.textContent='';
    if(!email||!pass){err.textContent='Vui lòng điền đầy đủ thông tin.';return;}
    try { loginUser(email,pass); hideAuthScreen(); updateTopbarUser(); }
    catch(e){err.textContent=e.message;}
  });

  // Register
  $('registerSubmit').addEventListener('click',()=>{
    const name=$('regName').value.trim(); const email=$('regEmail').value.trim();
    const pass=$('regPassword').value; const conf=$('regConfirm').value;
    const err=$('registerError'); err.textContent='';
    if(!name||!email||!pass||!conf){err.textContent='Vui lòng điền đầy đủ thông tin.';return;}
    if(pass.length<6){err.textContent='Mật khẩu phải có ít nhất 6 ký tự.';return;}
    if(pass!==conf){err.textContent='Mật khẩu xác nhận không khớp.';return;}
    try { registerUser(name,email,pass); hideAuthScreen(); updateTopbarUser(); }
    catch(e){err.textContent=e.message;}
  });

  // Password toggles
  document.querySelectorAll('.auth-pw-toggle').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const inp=$(btn.dataset.target); if(!inp)return;
      inp.type=inp.type==='password'?'text':'password';
    });
  });

  // New chat
  $('newChatBtn').addEventListener('click',()=>{
    state.chatStarted=false; state.messages=[]; state.activeSession=null;
    msgContainer.innerHTML=''; welcomeScr.style.display=''; msgContainer.style.display='none';
    buildHistory(); closeSidebar();
  });

  // Sidebar toggle
  $('sidebarToggle').addEventListener('click',()=>sidebar.classList.contains('open')?closeSidebar():openSidebar());
  overlay.addEventListener('click',()=>{closeSidebar(); hideUserDropdown();});

  // Topic chips
  document.querySelectorAll('.topic-chip').forEach(chip=>{
    chip.addEventListener('click',()=>{ textarea.value=chip.dataset.suggest; autoResize(textarea); sendBtn.disabled=false; textarea.focus(); closeSidebar(); });
  });

  // Suggestion cards
  document.querySelectorAll('.suggestion-card').forEach(card=>{
    card.addEventListener('click',()=>{ textarea.value=card.dataset.suggest; autoResize(textarea); sendBtn.disabled=false; sendMessage(); });
  });

  // Textarea
  textarea.addEventListener('input',()=>{ autoResize(textarea); sendBtn.disabled=!textarea.value.trim(); });
  textarea.addEventListener('keydown',e=>{ if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage();} });

  // Send
  sendBtn.addEventListener('click',sendMessage);

  // User dropdown
  $('topbarUserBtn').addEventListener('click',e=>{ e.stopPropagation(); userDropdown.classList.toggle('show'); });
  document.addEventListener('click',()=>hideUserDropdown());
  userDropdown.addEventListener('click',e=>e.stopPropagation());

  $('settingsBtn').addEventListener('click',()=>{ hideUserDropdown(); openSettings(); });
  $('settingsClose').addEventListener('click',closeSettings);
  settingsBackdrop.addEventListener('click',closeSettings);
  $('logoutBtn').addEventListener('click',()=>{ hideUserDropdown(); logoutUser(); });

  // User management
  $('userMgmtBtn').addEventListener('click',openUserMgmt);
  $('userMgmtClose').addEventListener('click',closeUserMgmt);
  userMgmtBackdrop.addEventListener('click',closeUserMgmt);
  $('addUserBtn').addEventListener('click',()=>showUserForm(null));
  $('userSearchInput').addEventListener('input',e=>buildUserList(e.target.value));

  // Theme opts
  document.querySelectorAll('.theme-opt').forEach(b=>{
    b.addEventListener('click',()=>applyTheme(b.dataset.themeVal));
  });
}

// ── Sidebar / Panels ──────────────────────────────────────────────────────────
function openSidebar()  { sidebar.classList.add('open'); overlay.classList.add('show'); }
function closeSidebar() { sidebar.classList.remove('open'); overlay.classList.remove('show'); }
function hideUserDropdown() { userDropdown.classList.remove('show'); }
function openSettings()  { settingsPanel.classList.add('open'); settingsBackdrop.classList.add('show'); }
function closeSettings() { settingsPanel.classList.remove('open'); settingsBackdrop.classList.remove('show'); }
function openUserMgmt()  { buildUserMgmtStats(); buildUserList(); userMgmtPanel.classList.add('open'); userMgmtBackdrop.classList.add('show'); }
function closeUserMgmt() { userMgmtPanel.classList.remove('open'); userMgmtBackdrop.classList.remove('show'); hideUserForm(); }

// ── History ───────────────────────────────────────────────────────────────────
function buildHistory() {
  const list=$('historyList'); if(!list)return; list.innerHTML='';
  if(!state.sessions.length){ list.innerHTML=`<div style="padding:16px 10px;font-size:12px;color:var(--text-muted);text-align:center;">Chưa có lịch sử hội thoại</div>`; return; }
  state.sessions.forEach(s=>{
    const item=document.createElement('div');
    item.className=`history-item${state.activeSession===s.id?' active':''}`;
    item.innerHTML=`<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg><span>${escapeHTML(s.title)}</span>`;
    item.addEventListener('click',()=>loadSession(s));
    list.appendChild(item);
  });
}

function loadSession(session) {
  state.activeSession=session.id; state.chatStarted=true;
  welcomeScr.style.display='none'; msgContainer.style.display='';
  msgContainer.innerHTML='';
  addMessage('user',`Cho tôi biết về: ${escapeHTML(session.title)}`);
  setTimeout(()=>callChatAPI(`Cho tôi biết về: ${session.title}`),300);
  buildHistory(); closeSidebar();
}

// ── Send ──────────────────────────────────────────────────────────────────────
function sendMessage() {
  const text=textarea.value.trim();
  if(!text||state.isTyping)return;
  if(!state.chatStarted){ state.chatStarted=true; welcomeScr.style.display='none'; msgContainer.style.display=''; }
  addMessage('user',escapeHTML(text));
  textarea.value=''; sendBtn.disabled=true; autoResize(textarea);
  callChatAPI(text);
}

// ── Chat backend API ──────────────────────────────────────────────────────────
// Backend FastAPI bọc pipeline RAG thật (Task 9 hybrid retrieval + Task 10
// generation có citation) — xem api/main.py. Đổi URL này thành Hugging Face
// Space sau khi deploy (xem .github/workflows/deploy-backend.yml).
const API_BASE = 'https://oggishi-lab08.hf.space';

async function callChatAPI(userText) {
  state.isTyping=true;
  const typingRow=showTyping();

  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ query: userText })
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    typingRow.remove();
    state.isTyping=false;

    if (data.answer) {
      addAIResponse(formatLegalResponse(data.answer) + formatSources(data.sources, data.retrieval_source), data.answer);
    } else {
      addAIResponse('<p style="color:var(--red)">Có lỗi xảy ra khi kết nối đến hệ thống. Vui lòng thử lại.</p>', '');
    }
  } catch(err) {
    typingRow.remove();
    state.isTyping=false;
    addAIResponse('<p style="color:var(--red)">Không thể kết nối đến hệ thống AI. Vui lòng kiểm tra kết nối mạng và thử lại.</p>', '');
  }
}

// Hiển thị các nguồn (chunks) mà pipeline dùng để tạo câu trả lời — Task 10
// được thiết kế riêng cho generation có citation nên hiển thị nguồn ở đây để
// người dùng đối chiếu, đúng tinh thần "trích dẫn minh bạch".
function formatSources(sources, retrievalSource) {
  if (!sources || !sources.length) return '';
  const chips = sources.map(s => {
    const label = escapeHTML(s.source || 'Nguồn');
    const score = (typeof s.score === 'number') ? ` · ${s.score.toFixed(2)}` : '';
    return `<span style="display:inline-block;padding:2px 8px;margin:2px;border:1px solid var(--border);border-radius:999px;font-family:var(--font-mono);font-size:10.5px;color:var(--text-muted);">${label}${score}</span>`;
  }).join('');
  return `<div style="margin-top:10px;font-size:11px;color:var(--text-muted);">
    <div style="margin-bottom:4px;">Nguồn tham khảo (${escapeHTML(retrievalSource || 'hybrid')}):</div>
    <div>${chips}</div>
  </div>`;
}

// ── Format legal response ─────────────────────────────────────────────────────
function formatLegalResponse(text) {
  // Convert markdown-like patterns to styled HTML
  let html = escapeHTML(text);

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Headers (## or ###)
  html = html.replace(/^###\s+(.+)$/gm, '<div style="font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.6px;color:var(--accent);margin:12px 0 5px;">$1</div>');
  html = html.replace(/^##\s+(.+)$/gm, '<div style="font-size:13px;font-weight:700;color:var(--text-primary);margin:10px 0 5px;border-bottom:1px solid var(--border);padding-bottom:4px;">$1</div>');

  // Law article citations (Điều XX)
  html = html.replace(/(Điều\s+\d+[a-z]?\s+(?:BLHS|Luật PCMT|Nghị định|Thông tư|NĐ)[^,;.\n]*)/gi,
    '<span class="law-cite" style="display:inline-block;font-family:var(--font-mono);font-size:11.5px;background:var(--bg-tertiary);border:1px solid var(--border);border-radius:4px;padding:1px 7px;margin:0 2px;color:var(--accent);">$1</span>');

  // Penalty keywords
  html = html.replace(/(phạt tù từ \d+.*?năm|tù chung thân|tử hình)/gi,
    '<span class="penalty-tag penalty-severe">$1</span>');
  html = html.replace(/(phạt tiền từ.*?đồng)/gi,
    '<span class="penalty-tag penalty-moderate">$1</span>');

  // Bullet lists
  html = html.replace(/^[-•]\s+(.+)$/gm,
    '<div style="display:flex;gap:8px;margin:3px 0;font-size:13px;line-height:1.65;"><span style="color:var(--accent);flex-shrink:0;margin-top:7px;width:5px;height:5px;border-radius:50%;background:var(--accent);display:inline-block;"></span><span>$1</span></div>');

  // Numbered lists
  html = html.replace(/^(\d+)\.\s+(.+)$/gm,
    '<div style="display:flex;gap:8px;margin:3px 0;font-size:13px;line-height:1.65;"><span style="font-family:var(--font-mono);font-size:11px;color:var(--accent);flex-shrink:0;min-width:16px;margin-top:1px;">$1.</span><span>$2</span></div>');

  // Paragraphs (wrap lines in <p>)
  html = html.split('\n\n').map(block => {
    block = block.trim();
    if (!block) return '';
    if (block.startsWith('<div') || block.startsWith('<span')) return block;
    return `<p style="margin:0 0 8px;font-size:13.5px;line-height:1.7;">${block}</p>`;
  }).join('');

  // Add disclaimer
  html += `<div class="disclaimer-note" style="margin-top:12px;">
    <span style="flex-shrink:0;">⚠</span>
    <span>Thông tin trên chỉ mang tính tham khảo và không thay thế tư vấn pháp lý chính thức. Đối với vụ việc cụ thể, hãy tham vấn luật sư hoặc cơ quan pháp luật có thẩm quyền.</span>
  </div>`;

  return html;
}

// ── Message rendering ─────────────────────────────────────────────────────────
function addMessage(role, html, rawText='') {
  const row=document.createElement('div');
  row.className=`message-row ${role}`;
  const now=new Date().toLocaleTimeString('vi-VN',{hour:'2-digit',minute:'2-digit'});
  const avatarInit=role==='ai'?'⚖':(state.currentUser?getInitials(state.currentUser.name):'U');
  const avatarCls =role==='ai'?'ai':`user ${state.currentUser?avatarClass(state.currentUser.role):''}`;
  row.innerHTML=`
    <div class="avatar ${avatarCls}">${avatarInit}</div>
    <div class="bubble-wrap">
      <div class="bubble ${role}">${html}</div>
      <div class="bubble-meta">
        <span class="msg-time">${now}</span>
        ${role==='ai'?`
          <button class="msg-action" onclick="copyBubble(this)">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Sao chép
          </button>
          <button class="msg-action" onclick="thumbsUp(this)">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
          </button>`:''
        }
      </div>
    </div>`;
  msgContainer.appendChild(row);
  scrollBottom();
  state.messages.push({role, content:html, rawText});
}

function addAIResponse(html, rawText) { addMessage('ai',html,rawText); }

function showTyping() {
  const row=document.createElement('div');
  row.className='message-row ai';
  row.innerHTML=`<div class="avatar ai">⚖</div><div class="bubble-wrap"><div class="bubble ai"><div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div></div></div>`;
  msgContainer.appendChild(row); scrollBottom(); return row;
}

// ── User management ───────────────────────────────────────────────────────────
function buildUserMgmtStats() {
  const users=loadUsers(); const admins=users.filter(u=>u.role==='admin').length; const active=users.filter(u=>u.status==='active').length;
  const el=$('userMgmtStats'); if(!el)return;
  el.innerHTML=`<div class="stat-card"><div class="stat-number">${users.length}</div><div class="stat-label">Tổng người dùng</div></div><div class="stat-card"><div class="stat-number">${admins}</div><div class="stat-label">Quản trị viên</div></div><div class="stat-card"><div class="stat-number">${active}</div><div class="stat-label">Đang hoạt động</div></div>`;
}

function buildUserList(filter='') {
  let users=loadUsers();
  if(filter){const q=filter.toLowerCase();users=users.filter(u=>u.name.toLowerCase().includes(q)||u.email.toLowerCase().includes(q));}
  const list=$('userList'); if(!list)return; list.innerHTML='';
  if(!users.length){list.innerHTML=`<div style="text-align:center;padding:32px;color:var(--text-muted);font-size:13px;">Không tìm thấy người dùng</div>`;return;}
  users.forEach(u=>{
    const isMe=state.currentUser&&u.id===state.currentUser.id;
    const card=document.createElement('div'); card.className='user-card'; card.dataset.uid=u.id;
    const roleName=u.role==='admin'?'Quản trị viên':u.role==='viewer'?'Xem':'Người dùng';
    card.innerHTML=`
      <div class="user-card-avatar ${avatarClass(u.role)}">${getInitials(u.name)}</div>
      <div class="user-card-info">
        <div class="user-card-name">${escapeHTML(u.name)}${isMe?` <span style="font-size:10px;color:var(--text-muted);">(bạn)</span>`:''}</div>
        <div class="user-card-meta-row"><div class="user-card-email">${escapeHTML(u.email)}</div><div class="status-dot${u.status==='inactive'?' inactive':''}"></div></div>
      </div>
      <div class="user-card-right">
        <span class="role-badge role-${u.role}">${roleName}</span>
        <div class="user-card-actions">
          <button class="user-action-btn" data-action="edit" data-uid="${u.id}" aria-label="Sửa"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg></button>
          <button class="user-action-btn danger" data-action="toggle" data-uid="${u.id}">${u.status==='active'?`<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="23" y1="11" x2="17" y2="11"/></svg>`:`<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/></svg>`}</button>
          <button class="user-action-btn danger" data-action="delete" data-uid="${u.id}" ${isMe?'disabled style="opacity:.4;cursor:not-allowed;"':''}><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg></button>
        </div>
      </div>`;
    list.appendChild(card);
  });
  list.querySelectorAll('.user-action-btn').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const uid=parseInt(btn.dataset.uid); const act=btn.dataset.action;
      if(act==='edit') showUserForm(uid);
      if(act==='toggle') toggleUserStatus(uid);
      if(act==='delete') deleteUser(uid,btn);
    });
  });
}

function showUserForm(uid=null) {
  state.userEditId=uid;
  const card=$('userFormCard'); if(!card)return;
  const user=uid?loadUsers().find(u=>u.id===uid):null;
  const isEdit=!!user;
  card.style.display='';
  card.innerHTML=`
    <div class="user-form-title"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">${isEdit?'<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/>':`<path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><line x1="20" y1="8" x2="20" y2="14"/><line x1="23" y1="11" x2="17" y2="11"/>`}</svg><span>${isEdit?'Sửa người dùng':'Thêm người dùng mới'}</span></div>
    <div class="user-form-grid">
      <div class="user-form-field full"><label class="user-form-label">Họ và tên</label><input class="user-form-input" id="uf_name" value="${isEdit?escapeHTML(user.name):''}" placeholder="Nguyễn Văn A"/></div>
      <div class="user-form-field full"><label class="user-form-label">Email</label><input class="user-form-input" id="uf_email" type="email" value="${isEdit?escapeHTML(user.email):''}" placeholder="email@example.com" ${isEdit?'readonly style="opacity:.6"':''}/></div>
      ${!isEdit?`<div class="user-form-field full"><label class="user-form-label">Mật khẩu</label><input class="user-form-input" id="uf_pass" type="password" placeholder="Tối thiểu 6 ký tự"/></div>`:''}
      <div class="user-form-field"><label class="user-form-label">Vai trò</label><select class="user-form-select" id="uf_role"><option value="user"${(!isEdit||user.role==='user')?' selected':''}>Người dùng</option><option value="admin"${isEdit&&user.role==='admin'?' selected':''}>Quản trị viên</option><option value="viewer"${isEdit&&user.role==='viewer'?' selected':''}>Xem</option></select></div>
      <div class="user-form-field"><label class="user-form-label">Trạng thái</label><select class="user-form-select" id="uf_status"><option value="active"${(!isEdit||user.status==='active')?' selected':''}>Hoạt động</option><option value="inactive"${isEdit&&user.status==='inactive'?' selected':''}>Vô hiệu hóa</option></select></div>
    </div>
    <div class="user-form-actions">
      <button class="btn-cancel-user" id="uf_cancel">Hủy</button>
      <button class="btn-save-user" id="uf_save">Lưu</button>
    </div>`;
  $('uf_cancel').addEventListener('click',hideUserForm);
  $('uf_save').addEventListener('click',()=>saveUserForm(uid));
}

function hideUserForm() { const c=$('userFormCard'); if(c)c.style.display='none'; }

function saveUserForm(uid) {
  const name=$('uf_name').value.trim(); const email=uid?null:$('uf_email').value.trim();
  const pass=uid?null:$('uf_pass')?.value; const role=$('uf_role').value; const status=$('uf_status').value;
  if(!name||(email!==null&&!email)||(pass!==null&&!pass)){alert('Vui lòng điền đầy đủ thông tin.');return;}
  const users=loadUsers();
  if(uid){const i=users.findIndex(u=>u.id===uid);if(i<0)return;users[i].name=name;users[i].role=role;users[i].status=status;}
  else{if(users.find(u=>u.email.toLowerCase()===email.toLowerCase())){alert('Email này đã được sử dụng.');return;}users.push({id:Date.now(),name,email,password:pass,role,status,joined:new Date().toISOString().slice(0,10)});}
  saveUsers(users); hideUserForm(); buildUserMgmtStats(); buildUserList($('userSearchInput').value);
  if(state.currentUser&&uid===state.currentUser.id){state.currentUser=users.find(u=>u.id===uid);updateTopbarUser();}
}

function toggleUserStatus(uid) {
  const users=loadUsers(); const i=users.findIndex(u=>u.id===uid); if(i<0)return;
  if(users[i].id===state.currentUser?.id){alert('Không thể vô hiệu hóa tài khoản của chính bạn.');return;}
  users[i].status=users[i].status==='active'?'inactive':'active';
  saveUsers(users); buildUserMgmtStats(); buildUserList($('userSearchInput').value);
}

function deleteUser(uid,btn) {
  if(!confirm('Bạn có chắc muốn xóa người dùng này?'))return;
  const users=loadUsers().filter(u=>u.id!==uid); saveUsers(users);
  buildUserMgmtStats(); buildUserList($('userSearchInput').value);
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function scrollBottom() { chatArea.scrollTop=chatArea.scrollHeight; }
function autoResize(el) { el.style.height='auto'; el.style.height=Math.min(el.scrollHeight,180)+'px'; }
function escapeHTML(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }

window.copyBubble = function(btn) {
  const bubble=btn.closest('.bubble-wrap').querySelector('.bubble');
  navigator.clipboard.writeText(bubble.innerText).then(()=>{
    btn.innerHTML=`<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Đã sao chép`;
    setTimeout(()=>{btn.innerHTML=`<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg> Sao chép`;},2000);
  });
};

window.thumbsUp = function(btn) { btn.style.color='var(--teal)'; };

document.addEventListener('DOMContentLoaded', init);
