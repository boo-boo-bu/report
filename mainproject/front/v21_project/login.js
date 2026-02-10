document.addEventListener('DOMContentLoaded', () => {
  const usernameInput = document.getElementById('username');
  const passwordInput = document.getElementById('password');
  const submitLogin = document.getElementById('submitLogin');
  const showSignupInline = document.getElementById('showSignupInline');
  const signupInline = document.getElementById('signupInline');
  const cancelSignup = document.getElementById('cancelSignup');
  const submitSignup = document.getElementById('submitSignup');
  const su_username = document.getElementById('su_username');
  const su_password = document.getElementById('su_password');
  const su_password_confirm = document.getElementById('su_password_confirm');

  const USERS_KEY = 'dashboard_users';
  const SESSION_KEY = 'dashboard_user';
  const ROLE_KEY = 'dashboard_role';

  function getUsers(){
    try{ return JSON.parse(localStorage.getItem(USERS_KEY) || '{}'); }catch(e){return {}};
  }
  function setUsers(u){ localStorage.setItem(USERS_KEY, JSON.stringify(u)); }

  function resetSignup(){
    su_username.value = '';
    su_password.value = '';
    su_password_confirm.value = '';
  }

  function resetLogin(){
    usernameInput.value = '';
    passwordInput.value = '';
  }

  function setRoleIfEmpty(){
    if(!localStorage.getItem(ROLE_KEY)) localStorage.setItem(ROLE_KEY, 'user');
  }

  function setRoleUser(){
    localStorage.setItem(ROLE_KEY, 'user');
  }

  // If already logged in, go to dashboard
  if(localStorage.getItem(SESSION_KEY)){
    window.location.href = 'monitoring.html';
    return;
  }

  showSignupInline.addEventListener('click', (e) => {
    e.preventDefault();
    signupInline.classList.toggle('hidden');
  });

  cancelSignup.addEventListener('click', () => {
    signupInline.classList.add('hidden');
    resetSignup();
  });

  submitSignup.addEventListener('click', () => {
    const u = su_username.value.trim();
    const p = su_password.value;
    const pc = su_password_confirm.value;
    if(!u){ alert('사용자명을 입력하세요'); return; }
    if(p.length < 4){ alert('비밀번호는 4자 이상이어야 합니다'); return; }
    if(p !== pc){ alert('비밀번호가 일치하지 않습니다'); return; }
    const users = getUsers();
    if(users[u]){ alert('이미 존재하는 사용자명입니다'); return; }
    users[u] = p; setUsers(users);
    localStorage.setItem(SESSION_KEY, u);
    setRoleUser();
    window.location.href = 'monitoring.html';
  });

  submitLogin.addEventListener('click', () => {
    const u = usernameInput.value.trim();
    const p = passwordInput.value;
    if(!u){ alert('사용자명을 입력하세요'); return; }
    const users = getUsers();
    if(!users[u]){ alert('존재하지 않는 사용자입니다. 회원가입 해주세요.'); return; }
    if(users[u] !== p){ alert('비밀번호가 틀렸습니다'); return; }
    localStorage.setItem(SESSION_KEY, u);
    setRoleIfEmpty();
    window.location.href = 'monitoring.html';
  });

  // Allow Enter to submit
  document.addEventListener('keydown', (e) => {
    if(e.key === 'Enter') submitLogin.click();
  });
});
