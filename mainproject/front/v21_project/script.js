document.addEventListener('DOMContentLoaded', () => {
  // localStorage keys
  const SESSION_KEY = 'dashboard_user';
  const ROLE_KEY = 'dashboard_role';

  // Redirect to index.html if not logged in
  if(!localStorage.getItem(SESSION_KEY)){
    window.location.href = 'index.html';
    return;
  }

  const userArea = document.getElementById('userArea');
  const roiControls = document.getElementById('roiControls');
  const roiStart = document.getElementById('roiStart');
  const roiReset = document.getElementById('roiReset');
  const roiDone = document.getElementById('roiDone');
  const roiLog = document.getElementById('roiLog');
  const roiInfo = document.getElementById('roiInfo');
  const videoFeed = document.getElementById('video-feed');
  const videoUploader = document.getElementById('videoUploader');
  const uploadTrigger = document.getElementById('uploadTrigger');
  const roleControls = document.getElementById('roleControls');
  const roleLabel = document.getElementById('roleLabel');
  const roleUserBtn = document.getElementById('roleUser');
  const roleAdminBtn = document.getElementById('roleAdmin');

  let roiMode = false;
  let roiPoints = [];

  function isLoggedIn(){
    return !!localStorage.getItem(SESSION_KEY);
  }

  function getRole(){
    return localStorage.getItem(ROLE_KEY) || 'user';
  }

  function setRole(role){
    localStorage.setItem(ROLE_KEY, role);
    updateRoleUI();
  }

  function setActiveNav(page){
    document.querySelectorAll('.sidebar a').forEach(x => x.classList.remove('active'));
    const link = document.querySelector(`.sidebar a[data-page="${page}"]`);
    if(link) link.classList.add('active');
  }

  function updateNavVisibility(){
    const logged = isLoggedIn();
    document.querySelectorAll('.sidebar a').forEach(a => {
      if(!logged && a.dataset.page !== 'overview'){
        a.classList.add('hidden');
      } else {
        a.classList.remove('hidden');
      }
    });
    if(!logged){
      showPage('overview');
      setActiveNav('overview');
    }
  }

  function updateRoleUI(){
    const logged = isLoggedIn();
    if(!logged){
      roleControls.classList.add('hidden');
      roiControls.classList.add('hidden');
      roiInfo.classList.add('hidden');
      return;
    }
    roleControls.classList.remove('hidden');
    const role = getRole();
    roleLabel.textContent = role === 'admin' ? '관리자' : '일반';
    if(role === 'admin'){
      roiControls.classList.remove('hidden');
      roiInfo.classList.remove('hidden');
    } else {
      roiControls.classList.add('hidden');
      roiInfo.classList.add('hidden');
    }
  }

  function renderUser(){
    const user = localStorage.getItem(SESSION_KEY);
    if(user){
      if(!localStorage.getItem(ROLE_KEY)) setRole('user');
      userArea.innerHTML = `<span style="color:#9fb4c9;margin-right:8px">${user}</span><button id='logoutBtn' class='btn'>로그아웃</button>`;
      document.getElementById('logoutBtn').addEventListener('click', () => {
        localStorage.removeItem(SESSION_KEY);
        // Keep ROLE_KEY to preserve user role across sessions
        window.location.href = 'index.html';
      });
    } else {
      userArea.innerHTML = `<button id='loginBtn' class='btn'>로그인</button>`;
      document.getElementById('loginBtn').addEventListener('click', () => { window.location.href = 'login.html'; });
    }
    updateNavVisibility();
    updateRoleUI();
  }

  // navigation
  document.querySelectorAll('.sidebar a').forEach(a => {
    a.addEventListener('click', (e) => {
      e.preventDefault();
      if(!isLoggedIn() && a.dataset.page !== 'overview') return;
      document.querySelectorAll('.sidebar a').forEach(x => x.classList.remove('active'));
      a.classList.add('active');
      showPage(a.dataset.page);
    });
  });

  function showPage(page){
    if(!isLoggedIn() && page !== 'overview') page = 'overview';
    document.querySelectorAll('.page').forEach(p => p.classList.add('hidden'));
    const el = document.getElementById(page);
    if(el) el.classList.remove('hidden');
    if(page === 'live'){
      document.querySelector('.status')?.classList.remove('hidden');
    } else {
      document.querySelector('.status')?.classList.add('hidden');
    }
  }

  function setRoiLog(){
    if(roiPoints.length === 0){
      roiLog.textContent = '(클릭 좌표가 여기에 표시됩니다)';
      return;
    }
    roiLog.textContent = roiPoints.map((p, i) => `#${i+1}  x:${p.x.toFixed(3)}  y:${p.y.toFixed(3)}  (px ${p.px}, ${p.py})`).join('\n');
  }

  roiStart.addEventListener('click', () => {
    if(getRole() !== 'admin') return;
    roiMode = true;
    roiStart.textContent = 'ROI 설정 중...';
  });

  roiReset.addEventListener('click', () => {
    if(getRole() !== 'admin') return;
    roiPoints = [];
    roiMode = false;
    roiStart.textContent = 'ROI 설정 시작';
    setRoiLog();
  });

  roiDone.addEventListener('click', () => {
    if(getRole() !== 'admin') return;
    roiMode = false;
    roiStart.textContent = 'ROI 설정 시작';
    // TODO: 서버 전송 로직 연결
    alert(`좌표 ${roiPoints.length}개 저장됨`);
  });

  videoFeed.addEventListener('click', (e) => {
    if(!isLoggedIn()) return;
    if(getRole() !== 'admin') return;
    if(!roiMode) return;
    const rect = videoFeed.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    roiPoints.push({ x, y, px: Math.round(e.clientX - rect.left), py: Math.round(e.clientY - rect.top) });
    setRoiLog();
  });

  let currentObjectUrl = null;

  videoUploader.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if(!file) return;
    if(currentObjectUrl){
      URL.revokeObjectURL(currentObjectUrl);
      currentObjectUrl = null;
    }
    const url = URL.createObjectURL(file);
    currentObjectUrl = url;
    videoFeed.src = url;
  });

  uploadTrigger.addEventListener('click', () => {
    // allow re-uploading the same file by resetting input value
    videoUploader.value = '';
    videoUploader.click();
  });


  roleUserBtn.addEventListener('click', () => setRole('user'));
  roleAdminBtn.addEventListener('click', () => setRole('admin'));

  // Playback: Add new violation card dynamically
  window.addViolationCard = function(data) {
    const grid = document.getElementById('playbackGrid');
    if(!grid) return;
    
    const card = document.createElement('div');
    card.className = 'playback-card';
    card.innerHTML = `
      <div class="playback-img">${data.image || '차량 이미지'}</div>
      <div class="playback-info">
        <div class="playback-title">${data.title || '위반 종류'}</div>
        <div class="playback-meta">차량번호: ${data.plateNumber || '-'}</div>
        <div class="playback-meta">시간: ${data.time || '-'}</div>
        <div class="playback-meta">위치: ${data.location || '-'}</div>
      </div>
    `;
    grid.appendChild(card);
  };

  // initial render
  showPage('live');
  renderUser();
  setRoiLog();
});
