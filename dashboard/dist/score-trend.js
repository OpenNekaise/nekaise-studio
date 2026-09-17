// Descriptive Gaussian smoothing of displayed observations, never a new measurement.
// Equal observation weights; bandwidth is 8% of extent, floored at 1.5 median x gaps.
export function smoothScores(points) {
  const valid = points.filter(p => p && Number.isFinite(p.x) && Number.isFinite(p.y) && p.y >= 0 && p.y <= 1);
  const positions = [...new Set(valid.map(p => p.x))].sort((a, b) => a - b);
  if (positions.length < 3) return [];
  const first = positions[0], span = positions.at(-1) - first;
  if (!Number.isFinite(span) || span <= 0) return [];
  const gaps = positions.slice(1).map((at, i) => (at - positions[i]) / span).sort((a, b) => a - b);
  const median = (gaps[Math.floor((gaps.length - 1) / 2)] + gaps[Math.floor(gaps.length / 2)]) / 2;
  const bandwidth = Math.max(.08, 1.5 * median);
  const samples = valid.map(p => ({ x: (p.x - first) / span, y: p.y }));
  return Array.from({ length: 81 }, (_, i) => {
    const at = i / 80;
    const distances = samples.map(p => ((p.x - at) / bandwidth) ** 2);
    const nearest = distances.reduce((a, b) => Math.min(a, b), Infinity);
    let total = 0, weighted = 0;
    for (let j = 0; j < samples.length; j++) {
      // Subtract the nearest squared distance to avoid underflow across sparse gaps.
      const weight = Math.exp(-.5 * (distances[j] - nearest));
      total += weight;
      weighted += weight * samples[j].y;
    }
    return { x: first + at * span, y: Math.max(0, Math.min(1, weighted / total)) };
  });
}

export function scoreTrend(points, x, y) {
  const fitted = smoothScores(points);
  if (!fitted.length) return "";
  const path = fitted.map((p, i) => `${i ? "L" : "M"}${x(p.x).toFixed(2)},${y(p.y).toFixed(2)}`).join(" ");
  return `<path class="score-trend" d="${path}"><title>Smoothed trend of displayed observations; endpoints are smoothed too</title></path>`;
}

export const trendLegend = '<div class="score-trend-legend"><span>Smoothed trend</span><span>Observed scores</span></div>';
