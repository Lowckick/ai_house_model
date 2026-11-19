"""Result figures for the README and docs, regenerated from the published JSON reports.

Each figure is written twice (``-light.png`` and ``-dark.png``) so GitHub can show the one that
matches the reader's theme. Colours are the validated reference palette (slots 1–4); every
figure has a table with the same numbers in docs/results.md.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
THEMES = {
    'light': {'surface': '#fcfcfb', 'text': '#0b0b0b', 'muted': '#52514e', 'grid': '#e1e0d9', 'axis': '#c3c2b7',
              'series': ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'], 'neutral': '#a3a29b'},
    'dark': {'surface': '#1a1a19', 'text': '#f0efec', 'muted': '#c3c2b7', 'grid': '#2c2c2a', 'axis': '#383835',
             'series': ['#3987e5', '#d95926', '#199e70', '#c98500'], 'neutral': '#6d6c66'},
}
HOUSES = {'default': 'Типовий', 'realistic_co2': 'Реалістичний\nCO₂', 'poor_insulation': 'Погана\nізоляція',
          'lagged_schedule': 'Інерційне\nопалення', 'combined': 'Усе\nразом'}
POLICIES = {'rules_default': 'Пороги, типові', 'rules_retuned': 'Пороги, перепідібрані',
            'nn_default': 'Мережа з типового будинку', 'nn_retrained': 'Мережа, перенавчена'}


def _mean(values):
    return sum(values) / len(values)


def model_rows():
    """Policies of the tariff experiment (rules_v2, both test sets averaged)."""
    report = json.loads((ROOT / 'data/models/rollout-v3-tou-evaluation.json').read_text())
    run = next(r for r in report['runs'] if r['seed'] == report['selected_seed'])
    refs = {Path(r['model']).name: r for r in report['references']}
    seeds = list(run['test'])

    def avg(get):
        return {m: _mean([get(s)[m] for s in seeds]) for m in ('avg_comfort', 'energy_kwh', 'cost_uah', 'unsafe_steps')}
    return [
        ('Перша DQN (model.pt)', False, avg(lambda s: refs['model.pt']['test'][s]['summary']['model'])),
        ('Простий контролер', False, avg(lambda s: run['test'][s]['summary']['deterministic'])),
        ('Посилений контролер', False, avg(lambda s: run['test'][s]['summary']['tuned'])),
        ('rollout-v2', True, avg(lambda s: refs['rollout-v2.pt']['test'][s]['summary']['model'])),
        ('rollout-v3-tou', True, avg(lambda s: run['test'][s]['summary']['model'])),
    ]


def adaptation_rows():
    report = json.loads((ROOT / 'docs/results/adaptation-2026-11-18.json').read_text())
    rows = []
    for house, label in HOUSES.items():
        tests = report['houses'][house]['tests']
        reward = {p: _mean([t['summary'][p]['total_reward'] for t in tests.values()]) for p in POLICIES}
        paired = [t['paired']['nn_retrained - rules_retuned'] for t in tests.values()]
        diff = {m: _mean([p[m]['mean'] for p in paired]) for m in ('total_reward', 'avg_comfort')}
        se = _mean([p['total_reward']['se'] ** 2 for p in paired]) ** 0.5 / len(paired) ** 0.5
        rows.append((label, reward, diff, se))
    return rows


def _style(ax, t, axis='x'):
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['axis'])
    ax.tick_params(colors=t['muted'], labelsize=9)
    ax.grid(axis=axis, color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)


def _figure(t, *args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor(t['surface'])
    return fig, axes


def figure_models(t):
    rows = model_rows()
    panels = (('avg_comfort', 'Комфорт, % ↑', '{:.1f}'), ('energy_kwh', 'Енергія, кВт·год/добу ↓', '{:.1f}'),
              ('cost_uah', 'Вартість, грн/добу ↓', '{:.0f}'), ('unsafe_steps', 'Небезпечні кроки/добу ↓', '{:.1f}'))
    fig, axes = _figure(t, 1, 4, figsize=(12, 3.4), sharey=True)
    names = [r[0] for r in rows][::-1]
    for ax, (metric, title, fmt) in zip(axes, panels):
        values = [r[2][metric] for r in rows][::-1]
        colors = [t['series'][0] if r[1] else t['neutral'] for r in rows][::-1]
        bars = ax.barh(names, values, color=colors, height=0.62, edgecolor=t['surface'], linewidth=2)
        ax.set_xlim(left=0)   # bars always start at zero
        for bar, value in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, ' ' + fmt.format(value),
                    va='center', fontsize=9, color=t['text'])
        ax.set_title(title, fontsize=10, color=t['text'], loc='left')
        _style(ax, t)
        ax.margins(x=0.18)
    axes[0].tick_params(axis='y', colors=t['text'], labelsize=9)
    fig.suptitle('Еволюція моделей: 40 test-сценаріїв, правила rules_v2, двозонний тариф',
                 x=0.01, ha='left', fontsize=11, color=t['text'])
    fig.text(0.01, -0.02, 'Синім — моделі rollout-навчання; сірим — базові рівні. Середнє за test-наборами seed 20000 і 31000.',
             fontsize=8.5, color=t['muted'])
    fig.tight_layout()
    return fig


def figure_adaptation(t):
    rows = adaptation_rows()
    fig, ax = _figure(t, figsize=(11, 4.2))
    width = 0.2
    for index, (key, label) in enumerate(POLICIES.items()):
        xs = [i + (index - 1.5) * width for i in range(len(rows))]
        ax.bar(xs, [r[1][key] for r in rows], width=width, color=t['series'][index], label=label,
               edgecolor=t['surface'], linewidth=2)
    ax.set_xticks(range(len(rows)), [r[0] for r in rows], color=t['text'], fontsize=9)
    ax.set_ylabel('Винагорода за добу ↑', color=t['muted'], fontsize=9)
    _style(ax, t, axis='y')
    ax.set_ylim(bottom=0)
    legend = ax.legend(ncol=4, frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.13))
    for text in legend.get_texts():
        text.set_color(t['text'])
    ax.set_title('Адаптація до п’яти будинків (40 test-сценаріїв на будинок)', fontsize=11, color=t['text'],
                 loc='left', pad=34)
    fig.tight_layout()
    return fig


def figure_advantage(t):
    rows = adaptation_rows()
    fig, ax = _figure(t, figsize=(8, 3.6))
    names = [r[0].replace('\n', ' ') for r in rows][::-1]
    values = [r[2]['total_reward'] for r in rows][::-1]
    errors = [r[3] for r in rows][::-1]
    comfort = [r[2]['avg_comfort'] for r in rows][::-1]
    bars = ax.barh(names, values, xerr=errors, color=t['series'][0], height=0.55, edgecolor=t['surface'], linewidth=2,
                   error_kw={'ecolor': t['muted'], 'elinewidth': 1, 'capsize': 3})
    for bar, value, error, c in zip(bars, values, errors, comfort):
        ax.text(value + error + 0.4, bar.get_y() + bar.get_height() / 2, f'+{value:.1f}  (комфорт +{c:.1f} в.п.)',
                va='center', fontsize=9, color=t['text'])
    ax.axvline(0, color=t['axis'], linewidth=1)
    ax.set_xlim(0, max(v + e for v, e in zip(values, errors)) * 1.75)
    ax.tick_params(axis='y', colors=t['text'])
    ax.set_xlabel('Різниця винагороди за добу (± стандартна похибка)', color=t['muted'], fontsize=9)
    _style(ax, t)
    ax.set_title('Перенавчена мережа мінус перепідібрані правила', fontsize=11, color=t['text'], loc='left')
    fig.tight_layout()
    return fig


FIGURES = {'models': figure_models, 'adaptation': figure_adaptation, 'advantage': figure_advantage}


def generate(output_dir=ROOT / 'docs' / 'images'):
    """Write every figure in both themes; returns the paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, build in FIGURES.items():
        for theme, tokens in THEMES.items():
            plt.rcParams['font.family'] = 'DejaVu Sans'
            fig = build(tokens)
            path = output_dir / f'{name}-{theme}.png'
            fig.savefig(path, dpi=160, bbox_inches='tight', facecolor=tokens['surface'])
            plt.close(fig)
            paths.append(path)
    return paths
