/**
 * ModelInfo — Informational tab about the Dixon-Coles model,
 * data sources, and features used for predictions.
 */
function ModelInfo() {
    return (
        <div className="space-y-6 animate-fade-in">
            {/* Model Overview */}
            <section className="glass-card p-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500/20 to-blue-600/10 flex items-center justify-center border border-blue-500/20">
                        <span className="text-lg">🧮</span>
                    </div>
                    <h2 className="text-lg font-bold text-white">Modelo Dixon-Coles (1997)</h2>
                </div>
                <p className="text-sm text-slate-400 leading-relaxed mb-4">
                    El modelo Dixon-Coles es una extensión del modelo de Poisson bivariado para predicción de
                    resultados de fútbol. Fue propuesto por Mark Dixon y Stuart Coles en su paper
                    <em className="text-slate-300">"Modelling Association Football Scores and Inefficiencies in the Football Betting Market"</em> (1997).
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-white/[0.03] rounded-xl p-4 border border-white/5">
                        <h3 className="text-sm font-semibold text-blue-400 mb-2">📐 Formulación Matemática</h3>
                        <div className="space-y-2 text-xs text-slate-400 font-mono">
                            <p>λ<sub>home</sub> = exp(α<sub>i</sub> + β<sub>j</sub> + γ)</p>
                            <p>μ<sub>away</sub> = exp(α<sub>j</sub> + β<sub>i</sub>)</p>
                            <p className="text-slate-500 mt-2 font-sans">Donde:</p>
                            <ul className="list-disc list-inside space-y-1 font-sans">
                                <li><span className="font-mono text-emerald-400">α</span> = parámetro de ataque del equipo</li>
                                <li><span className="font-mono text-rose-400">β</span> = parámetro de defensa del equipo</li>
                                <li><span className="font-mono text-amber-400">γ</span> = ventaja de local (home advantage)</li>
                            </ul>
                        </div>
                    </div>
                    <div className="bg-white/[0.03] rounded-xl p-4 border border-white/5">
                        <h3 className="text-sm font-semibold text-purple-400 mb-2">🔧 Corrección Rho (ρ)</h3>
                        <p className="text-xs text-slate-400 leading-relaxed">
                            La innovación clave del modelo es la <strong className="text-slate-300">corrección ρ</strong> que
                            ajusta las probabilidades para marcadores bajos (0-0, 0-1, 1-0, 1-1), corrigiendo la
                            tendencia del modelo Poisson puro a subestimar empates a cero y resultados de 1-0.
                        </p>
                        <div className="mt-3 flex items-center gap-2">
                            <span className="text-xs px-2 py-1 rounded-md bg-purple-500/10 text-purple-400 border border-purple-500/20 font-mono">
                                ρ ∈ [-1.5, 1.5]
                            </span>
                            <span className="text-xs text-slate-500">← rango permitido</span>
                        </div>
                    </div>
                </div>
            </section>

            {/* Features & Parameters */}
            <section className="glass-card p-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500/20 to-emerald-600/10 flex items-center justify-center border border-emerald-500/20">
                        <span className="text-lg">⚙️</span>
                    </div>
                    <h2 className="text-lg font-bold text-white">Parámetros y Features</h2>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                    {[
                        { icon: '⚔️', name: 'Ataque (α)', desc: 'Capacidad ofensiva de cada equipo, estimada a partir de goles anotados.' },
                        { icon: '🛡️', name: 'Defensa (β)', desc: 'Solidez defensiva de cada equipo, estimada a partir de goles recibidos.' },
                        { icon: '🏟️', name: 'Ventaja Local (γ)', desc: 'Boost multiplicativo para el equipo local. Típicamente +10-30% en odds.' },
                        { icon: '📊', name: 'Rho (ρ)', desc: 'Corrección de correlación para marcadores bajos. Valor negativo = más empates 0-0.' },
                        { icon: '⏳', name: 'Xi (ξ) — Time Decay', desc: 'Ponderación temporal. Partidos recientes pesan más que los antiguos.' },
                        { icon: '🎯', name: 'Top 3 Marcadores', desc: 'Se calculan las probabilidades de cada marcador posible (0-0 hasta 6-6).' },
                    ].map((feature, i) => (
                        <div key={i} className="bg-white/[0.03] rounded-xl p-4 border border-white/5 hover:border-emerald-500/20 transition-all duration-300">
                            <div className="flex items-center gap-2 mb-2">
                                <span>{feature.icon}</span>
                                <h4 className="text-sm font-semibold text-white">{feature.name}</h4>
                            </div>
                            <p className="text-xs text-slate-400 leading-relaxed">{feature.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* Data Sources */}
            <section className="glass-card p-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500/20 to-amber-600/10 flex items-center justify-center border border-amber-500/20">
                        <span className="text-lg">📦</span>
                    </div>
                    <h2 className="text-lg font-bold text-white">Fuentes de Datos</h2>
                </div>
                <div className="space-y-3">
                    {[
                        { source: 'FBref (soccerdata)', type: 'Primaria', desc: 'Estadísticas históricas de partidos de las principales ligas: Liga MX, Premier League, La Liga, Serie A, Ligue 1, Bundesliga.', color: 'emerald' },
                        { source: 'Web Search Fallback', type: 'Secundaria', desc: 'Búsqueda automática de resultados recientes cuando un equipo no está en FBref (ej: ligas menores, Eredivisie, Belgian).', color: 'blue' },
                        { source: 'Dataset Sintético', type: 'Baseline', desc: 'Datos generados con fortalezas realistas por equipo. Se usa como base cuando las fuentes externas no están disponibles.', color: 'purple' },
                        { source: 'Lotería Nacional', type: 'Fixtures', desc: 'Scraping semanal de la quiniela oficial de Progol y Revancha desde loterianacional.gob.mx.', color: 'amber' },
                    ].map((src, i) => (
                        <div key={i} className="flex items-start gap-4 bg-white/[0.02] rounded-xl p-4 border border-white/5">
                            <span className={`text-xs px-2 py-1 rounded-md bg-${src.color}-500/10 text-${src.color}-400 border border-${src.color}-500/20 font-semibold whitespace-nowrap`}>
                                {src.type}
                            </span>
                            <div>
                                <h4 className="text-sm font-semibold text-white">{src.source}</h4>
                                <p className="text-xs text-slate-400 mt-1 leading-relaxed">{src.desc}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Pipeline */}
            <section className="glass-card p-6">
                <div className="flex items-center gap-3 mb-4">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-rose-500/20 to-rose-600/10 flex items-center justify-center border border-rose-500/20">
                        <span className="text-lg">🔄</span>
                    </div>
                    <h2 className="text-lg font-bold text-white">Pipeline de Predicción</h2>
                </div>
                <div className="flex flex-col gap-1">
                    {[
                        { step: 1, title: 'Scraping Quiniela', desc: 'Se obtienen los 14+7 partidos de la semana desde Lotería Nacional', icon: '🌐' },
                        { step: 2, title: 'Construcción del Dataset', desc: 'Se combinan datos históricos de FBref + fallback + sintéticos', icon: '📊' },
                        { step: 3, title: 'Ajuste del Modelo', desc: 'Optimización SLSQP del log-likelihood negativo (vectorizado, ~3s)', icon: '⚡' },
                        { step: 4, title: 'Generación de Predicciones', desc: 'Cálculo de P(L), P(E), P(V) y top 3 marcadores por partido', icon: '🎯' },
                        { step: 5, title: 'Cache y API', desc: 'Resultados servidos via FastAPI. Scheduler: lunes 09:00 CST', icon: '🚀' },
                    ].map((s, i) => (
                        <div key={i} className="flex items-start gap-4 relative">
                            {/* Connector line */}
                            {i < 4 && (
                                <div className="absolute left-5 top-12 w-0.5 h-6 bg-gradient-to-b from-white/10 to-transparent"></div>
                            )}
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-white/10 to-white/5 flex items-center justify-center border border-white/10 flex-shrink-0">
                                <span className="text-sm">{s.icon}</span>
                            </div>
                            <div className="pb-4">
                                <div className="flex items-center gap-2">
                                    <span className="text-[10px] font-mono text-slate-500">PASO {s.step}</span>
                                    <h4 className="text-sm font-semibold text-white">{s.title}</h4>
                                </div>
                                <p className="text-xs text-slate-400 mt-0.5">{s.desc}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Optimization note */}
            <section className="glass-card p-6">
                <div className="flex items-center gap-3 mb-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500/20 to-cyan-600/10 flex items-center justify-center border border-cyan-500/20">
                        <span className="text-lg">⚡</span>
                    </div>
                    <h2 className="text-lg font-bold text-white">Optimización Vectorizada</h2>
                </div>
                <p className="text-sm text-slate-400 leading-relaxed mb-3">
                    El cálculo del log-likelihood negativo está completamente vectorizado con <strong className="text-cyan-400">NumPy</strong>,
                    eliminando loops iterativos. Esto permite ajustar el modelo con ~1,700 partidos y 73 equipos en
                    <strong className="text-cyan-400"> ~3 segundos</strong>.
                </p>
                <div className="flex items-center gap-3 text-xs">
                    <span className="px-3 py-1.5 rounded-lg bg-red-500/10 text-red-400 border border-red-500/20 line-through">
                        iterrows() — minutos
                    </span>
                    <span className="text-slate-500">→</span>
                    <span className="px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        np.array — ~3 segundos ⚡
                    </span>
                </div>
            </section>

            {/* Reference */}
            <div className="text-center py-4">
                <p className="text-xs text-slate-600">
                    Dixon, M. J. & Coles, S. G. (1997). "Modelling Association Football Scores and Inefficiencies
                    in the Football Betting Market." <em>Journal of the Royal Statistical Society: Series C</em>, 46(2), 265-280.
                </p>
            </div>
        </div>
    )
}

export default ModelInfo
