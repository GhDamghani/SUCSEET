def plot(lbl, melSpec, original_audio, N, ratio):
    fig, ax = plt.subplots(3, 1, figsize=(5, 5))
    offset = 3130
    plot_update(lbl, melSpec, original_audio, N, ax, offset, ratio)
    slider_ax = fig.add_axes([0.2, 0, 0.6, 0.03])
    slider = Slider(slider_ax, "X Offset", 0, lbl.size - N, valinit=0)

    def update(val):
        # Get the current slider position
        offset = int(slider.val)
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        # Create a new view of the image with the desired offset
        plot_update(lbl, melSpec, original_audio, N, ax, offset, ratio)
        plt.draw()

    # slider.on_changed(update)
    plt.tight_layout()
    plt.savefig("demo.png", transparent=True)


color_pallet = [
    "#808080",
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
    "#fb9a99",
    "#e31a1c",
    "#fdbf6f",
    "#ff7f00",
    "#cab2d6",
    "#6a3d9a",
    "#ffff99",
    "#b15928",
    "#8dd3c7",
    "#ffffb3",
    "#bebada",
    "#fb8072",
    "#80b1d3",
    "#fdb462",
    "#b3de69",
    "#fccde5",
    "#d9d9d9",
    "#bc80bd",
    "#ccebc5",
    "#ffed6f",
]
color_pallet = [
    (int(x[1:3], 16) / 255, int(x[3:5], 16) / 255, int(x[5:7], 16) / 255)
    for x in color_pallet
]


def plot_update(lbl, melSpec, original_audio, N, ax, offset, ratio):
    class_image = np.zeros((1, N, 3))
    for i in range(N):
        class_image[0, i] = color_pallet[lbl[offset + i]]
    ax[0].plot(original_audio[offset * ratio : (offset + N) * ratio])
    ax[0].set_xlim(0, N * ratio)
    ax[0].set_ylim(-1.0, 1.0)
    ax[0].axis("off")
    ax[1].imshow(melSpec[offset : offset + N].T, "gray", aspect="auto")
    ax[1].axis("off")
    ax[2].imshow(class_image, aspect="auto")
    ax[2].axis("off")


def plot_hist(lbl):
    lbl_hist = np.unique(lbl, return_counts=True)
    plt.figure()
    plt.stem(
        lbl_hist[0],
        lbl_hist[1] / np.sum(lbl_hist[1]),
    )
    plt.xlabel("Labels")
    plt.ylabel("Freq")
    plt.title("Histogram of Clusters for the whole data")

    plt.tight_layout()
    return lbl_hist[1]


if plot_figure:
    fold = 0
    original_audio = scipy.io.wavfile.read(
        join(path_input, f"{participant}_orig_synthesized.wav")
    )
    original_audio = original_audio[1] / 2**15
    ratio = original_audio.shape[0] // X_whole.shape[0]

    plot(y[fold], X_whole, original_audio, 400, ratio)
    hist = plot_hist(y[fold])
    np.save(
        join(
            output_path,
            f"{participant}_spec_{vocoder_name}_cluster_{n_clusters:02d}_fold_{fold:02d}_hist.npy",
        ),
        hist,
    )
    plt.savefig(
        join(
            output_path,
            f"{participant}_spec_{vocoder_name}_cluster_{n_clusters:02d}_fold_{fold:02d}_hist.png",
        )
    )
