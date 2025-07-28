import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons
import numpy as np
from matplotlib.transforms import blended_transform_factory

def interactive_plot(generate_strategy_func, simulate_func, target_altitude):
    """
    Renders an interactive plot with a dropdown menu for different strategy profiles,
    and shows final stats like climb time and final speed.
    """
    profiles = [
        'linear',
        'exponential_increasing_climb',
        'exponential_decreasing_climb',
        'exponential_increasing_speed',
        'exponential_decreasing_speed',
        'constant_speed'
    ]

    fig = plt.figure(figsize=(14, 10), facecolor='#f5f5f5')
    axs = [plt.subplot2grid((10, 10), (0, 2), rowspan=4, colspan=8),
           plt.subplot2grid((10, 10), (5, 2), rowspan=4, colspan=8)]
    plt.subplots_adjust(left=0.3, right=0.95, top=0.92, bottom=0.1, hspace=0.4)

    radio_ax = plt.axes([0.05, 0.4, 0.2, 0.4], facecolor='#ebebeb')
    radio = RadioButtons(radio_ax, profiles, active=0)

    for label in radio.labels:
        label.set_color('darkblue')
        label.set_fontsize(10)

    stats_ax = plt.axes([0.05, 0.05, 0.25, 0.25], facecolor='#f0f8ff')
    stats_ax.set_title("Performance Summary", fontsize=11, pad=10, color='darkblue')
    stats_ax.axis('off')
    stats_text = stats_ax.text(0.5, 0.95, "", fontsize=9,
                               va='top', ha='center', transform=stats_ax.transAxes,
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def update_plot(selected_profile):
        for ax in axs:
            ax.clear()

        stats_lines = []
        strategies = generate_strategy_func(profile=selected_profile)
        is_constant_rate = selected_profile == 'constant_speed'

        colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(strategies))))
        line_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (3, 1, 1, 1)), (0, (5, 2))] * 5

        max_time = 0  # Track the longest simulation time

        for i, entry in enumerate(strategies):
            if is_constant_rate:
                _, strategy_function = entry
                label_text = "Constant Rate"
            else:
                altitude_fraction, strategy_function = entry
                label_text = f"AF: {altitude_fraction:.2f}"

            t, h, V = simulate_func(strategy_function)
            max_time = max(max_time, t[-1])  # Track maximum time

            style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]

            axs[0].plot(t, h, label=label_text,
                        color=color, linestyle=style, linewidth=2)
            axs[1].plot(t, V, label=label_text,
                        color=color, linestyle=style, linewidth=2)

            stats_lines.append(
                f"{label_text:>15}: Time: {t[-1]:6.1f}s | Speed: {V[-1]:6.1f}m/s"
            )

        formatted_title = selected_profile.replace('_', ' ').title()

        axs[0].set_title(f"Altitude Profile: {formatted_title} Strategy", fontsize=12, pad=12)
        axs[0].set_ylabel("Altitude (m)", fontsize=10)
        axs[0].axhline(target_altitude, color='#e41a1c', linestyle='--', linewidth=1.5, alpha=0.8)

        trans = blended_transform_factory(axs[0].transAxes, axs[0].transData)
        axs[0].text(1.02, target_altitude, ' Target Altitude',
                    transform=trans,
                    color='#e41a1c', va='center', ha='left', fontsize=9)

        axs[0].grid(True, linestyle=':', alpha=0.7)
        axs[0].legend(loc='best', frameon=True, framealpha=0.9, fontsize=9)

        axs[1].set_title(f"Speed Profile: {formatted_title} Strategy", fontsize=12, pad=12)
        axs[1].set_xlabel("Time (s)", fontsize=10)
        axs[1].set_ylabel("Speed (m/s)", fontsize=10)
        axs[1].grid(True, linestyle=':', alpha=0.7)
        axs[1].legend(loc='best', frameon=True, framealpha=0.9, fontsize=9)

        stats_text.set_text("\n".join(stats_lines))

        for ax in axs:
            ax.set_xlim(0, max_time * 1.05)
            for spine in ax.spines.values():
                spine.set_edgecolor('#cccccc')
                spine.set_linewidth(1.2)

        fig.canvas.draw_idle()

    radio.on_clicked(update_plot)
    update_plot(profiles[0])

    fig.suptitle("Aircraft Climb Profile Analysis", fontsize=14, fontweight='bold', y=0.995)
    plt.show()
