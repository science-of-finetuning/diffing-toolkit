// Detect Streamlit theme from URL query params
function detectTheme() {
  const params = new URLSearchParams(window.location.search);
  const bgColor = params.get('backgroundColor');
  if (bgColor) {
    // Parse the background color to determine if it's dark
    // Streamlit passes colors like "#0e1117" for dark mode
    const rgb = parseInt(bgColor.slice(1), 16);
    const r = (rgb >> 16) & 255;
    const g = (rgb >> 8) & 255;
    const b = rgb & 255;
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return luminance < 0.5 ? 'dark' : 'light';
  }
  // Fallback to system preference
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

// Sample cycling logic - manages showing/hiding samples and updating counter
function initSampleCycler(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // Apply theme class
  if (detectTheme() === 'dark') {
    container.classList.add('dark-theme');
  }

  const samples = container.querySelectorAll('.sample-content');
  const counter = container.querySelector('.sample-counter');
  const prevBtn = container.querySelector('.prev-btn');
  const nextBtn = container.querySelector('.next-btn');
  
  let currentIdx = 0;
  const total = samples.length;

  function updateDisplay() {
    samples.forEach((sample, i) => {
      sample.style.display = i === currentIdx ? 'block' : 'none';
    });
    if (counter) counter.textContent = `Sample ${currentIdx + 1} of ${total}`;
  }

  if (prevBtn) {
    prevBtn.addEventListener('click', () => {
      // Cycle back: if at 0, go to last
      currentIdx = (currentIdx - 1 + total) % total;
      updateDisplay();
    });
  }

  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      // Cycle forward: if at last, go to 0
      currentIdx = (currentIdx + 1) % total;
      updateDisplay();
    });
  }

  updateDisplay();
}
