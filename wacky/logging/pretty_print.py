class PrettyPrinter:
    def __init__(self):
        self.sections = {}
        self.max_key_len = 0
        self.max_value_len = 0

    def add_section(self, section_name, metrics=None):
        if metrics:
            for key, value in metrics.items():
                self.add_metric(section_name, key, value)
        else:
            self.sections[section_name] = {}

    def add_metric(self, section_name, metric_name, value):
        if section_name not in self.sections:
            self.sections[section_name] = {}
        self.sections[section_name][metric_name] = value
        self.max_key_len = max(self.max_key_len, len(metric_name))
        self.max_value_len = max(self.max_value_len, len(str(value)))

    def dump(self):
        total_length = self.max_key_len + self.max_value_len + 7
        print(' ' + "-" * (total_length + 1))
        for section, metrics in self.sections.items():
            print(f"| {section:<{total_length - 1}} |")
            for key, value in metrics.items():
                print(f"|    {key:<{self.max_key_len}} | {value:>{self.max_value_len}} |")
        print(' ' + "-" * (total_length + 1))
        self.sections.clear()
        self.max_key_len = 0
        self.max_value_len = 0


if __name__ == '__main__':
    # Example usage:
    pretty_printer = PrettyPrinter()
    pretty_printer.add_section("rollout/")
    pretty_printer.add_metric("rollout/", "ep_len_mean", 86.6)
    pretty_printer.add_metric("rollout/", "ep_rew_mean", -174)
    pretty_printer.add_section("time/")
    pretty_printer.add_metric("time/", "fps", 583)
    pretty_printer.add_metric("time/", "iterations", 2)
    pretty_printer.add_section("train/")
    pretty_printer.add_metric("train/", "approx_kl", 0.003588578989)
    pretty_printer.add_metric("train/", "clip_fraction", 0.00278)
    pretty_printer.dump()
