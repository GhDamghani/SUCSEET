from os.path import join


class Names:
    def __init__(
        self,
        output_path,
        dataset_name,
        vocoder_name,
        participant,
        nfolds,
        num_classes,
        clustering_method,
    ) -> None:
        self.output_path = output_path
        self.local_template = f"{dataset_name}_{vocoder_name}_{participant}_cluster_{clustering_method}_c{num_classes:02d}_f{nfolds:02d}"

    def update(self, fold):
        self.local = self.local_template + (
            f"_{fold:02d}" if isinstance(fold, int) else f"_{fold}"
        )
        self.file = join(
            self.output_path,
            self.local,
        )
        self.model = join(
            self.output_path,
            self.local + "_model.pth",
        )
        self.results = join(
            self.output_path,
            self.local + "_results",
        )
        self.stats = join(
            self.output_path,
            self.local + "_stats",
        )
        self.confusion = join(
            self.output_path,
            self.local + "_confusion.png",
        )
        self.histogram = join(
            self.output_path,
            self.local + "_histogram.png",
        )
