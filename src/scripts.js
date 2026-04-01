// ── Sidebar toggle (mobile) ──────────────────────────────────
function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('open');
}
document.addEventListener('click', e => {
  const sb = document.getElementById('sidebar');
  const hb = document.getElementById('hamburger');
  if (!sb.contains(e.target) && !hb.contains(e.target)) {
    sb.classList.remove('open');
  }
});

// ── Day accordion ────────────────────────────────────────────
function toggleDay(n) {
  const btn = document.querySelector(`#dg-${n} .day-toggle`);
  const sessions = document.getElementById(`ds-${n}`);
  const isOpen = sessions.classList.contains('open');
  btn.classList.toggle('open', !isOpen);
  sessions.classList.toggle('open', !isOpen);
}

// ── Scroll to section ────────────────────────────────────────
function scrollToSession(id) {
  const el = document.getElementById(id);
  if (el) {
    el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    document.querySelectorAll('.nav-session-btn').forEach(b => b.classList.remove('active'));
    const btns = document.querySelectorAll('.nav-session-btn');
    btns.forEach(b => {
      if (b.getAttribute('onclick') === `scrollToSession('${id}')`) b.classList.add('active');
    });
  }
  if (window.innerWidth < 900) document.getElementById('sidebar').classList.remove('open');
}

function scrollToTopic(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.scrollIntoView({ behavior: 'smooth', block: 'start' });
  // Update active topic highlight
  document.querySelectorAll('.nav-topic-link').forEach(l => l.classList.remove('active-topic'));
  const link = document.querySelector(`.nav-topic-link[data-topic='${id}']`);
  if (link) link.classList.add('active-topic');
  if (window.innerWidth < 900) document.getElementById('sidebar').classList.remove('open');
}

// ── Progress bar ─────────────────────────────────────────────
const progressBar = document.getElementById('progress');
const backTop = document.getElementById('back-top');
window.addEventListener('scroll', () => {
  const h = document.documentElement;
  const denom = h.scrollHeight - h.clientHeight; const prog = denom > 0 ? h.scrollTop / denom : 0;
  progressBar.style.transform = `scaleX(${prog})`;
  backTop.classList.toggle('visible', h.scrollTop > 400);
}, { passive: true });

// ── Active nav highlight via IntersectionObserver ────────────
const sessions = ['s11','s12','s13','s21','s22','s23','s31','s32','s33'];
const dayMap = { s11:1, s12:1, s13:1, s21:2, s22:2, s23:2, s31:3, s32:3, s33:3 };

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      const id = entry.target.id;
      // Update session buttons
      document.querySelectorAll('.nav-session-btn').forEach(b => b.classList.remove('active'));
      const match = document.querySelector(`.nav-session-btn[onclick="scrollToSession('${id}')"]`);
      if (match) match.classList.add('active');
      // Open correct day
      const day = dayMap[id];
      if (day) {
        [1,2,3].forEach(d => {
          const active = d === day;
          const btn = document.querySelector(`#dg-${d} .day-toggle`);
          const ds = document.getElementById(`ds-${d}`);
          if (active) {
            btn?.classList.add('open','active-day');
            ds?.classList.add('open');
          } else {
            btn?.classList.remove('active-day');
          }
        });
      }
    }
  });
}, { rootMargin: '-20% 0px -60% 0px', threshold: 0 });

sessions.forEach(id => {
  const el = document.getElementById(id);
  if (el) observer.observe(el);
});

// ── Active topic links ───────────────────────────────────────
const topicEls = document.querySelectorAll('[id^="t-"]');
const topicObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      document.querySelectorAll('.nav-topic-link').forEach(l => l.classList.remove('active-topic'));
      const link = document.querySelector(`.nav-topic-link[data-topic='${entry.target.id}']`);
      if (link) link.classList.add('active-topic');
    }
  });
}, { rootMargin: '-10% 0px -70% 0px' });
topicEls.forEach(el => topicObserver.observe(el));

// ── Copy code ────────────────────────────────────────────────
function copyCode(btn) {
  const pre = btn.closest('.code-block').querySelector('pre');
  const text = pre.textContent;
  // Clipboard API may be blocked in sandboxed iframes — use execCommand fallback
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => flashCopied(btn)).catch(() => fallbackCopy(text, btn));
  } else {
    fallbackCopy(text, btn);
  }
}
function fallbackCopy(text, btn) {
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;opacity:0;top:0;left:0';
    document.body.appendChild(ta);
    ta.focus(); ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
    flashCopied(btn);
  } catch(e) {
    btn.textContent = 'error';
    setTimeout(() => { btn.textContent = 'copy'; }, 1800);
  }
}
function flashCopied(btn) {
  btn.textContent = 'copied!';
  btn.style.color = '#6ee7b7';
  setTimeout(() => { btn.textContent = 'copy'; btn.style.color = ''; }, 1800);
}

// ── Fade-up on scroll ─────────────────────────────────────────
const fadeEls = document.querySelectorAll('.session, .day-banner, .topic, .lab-block');
const fadeObs = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.animation = 'fadeUp 0.45s ease both';
      fadeObs.unobserve(e.target);
    }
  });
}, { threshold: 0.04 });
fadeEls.forEach(el => fadeObs.observe(el));
