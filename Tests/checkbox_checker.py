class CheckboxChecker:
    def __init__(self, checkbox, label):
        self.checkbox = checkbox
        self.label = label

    def is_checked(self):
        return self.checkbox.isChecked()
