from PIL import Image
from pathlib import Path
import numpy as np
import shutil

class Preprocessor:
    def __init__(self):
        self.BASE_PATH = "../data/"

    # Resize image. Maintain aspect ratio
    def resize(self, fp):
        im = Image.open(fp).convert("RGB")

        # Recalc scaled
        w, h = im.size
        scale = self.IMG_SIZE / min(w, h)
        w, h = int(w * scale), int(h * scale)

        # Scale
        im = im.resize((w, h), Image.LANCZOS)

        # Re-crop
        l = (w - self.IMG_SIZE) // 2
        t = (h - self.IMG_SIZE) // 2
        r = l + self.IMG_SIZE
        b = t + self.IMG_SIZE

        im = im.crop((l, t, r, b))

        return im

    def process_folder(self, fp_in, fp_out, label):
        fp_in = Path(fp_in)
        fp_out = Path(fp_out) / label  # Adds "academic" or "non_academic"
        fp_out.mkdir(parents=True, exist_ok=True)

        # Limit valid ext
        valid_ext = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

        # Loop all files in dir w/ valid ext
        im_fp = [f for f in fp_in.iterdir() if f.suffix in valid_ext]

        n_processed = 0
        fails = []

        for im in im_fp:
            try:
                resized = self.resize(im)

                out = fp_out / f"{im.stem}.jpeg"
                resized.save(out, "JPEG", quality=95)

                n_processed += 1

            except Exception as e:
                print(f"Failed to process {im.name}: {e}")
                fails.append(im.name)

        print(f"{label}: Sucessfully processed {n_processed} images")

        if fails:
            print(f"Failed to process {len(fails)} images")

        return n_processed

    def split(self, fp_in, fp_out, train_ratio, valid_ratio, test_ratio):
        assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

        fp_in = Path(fp_in)
        fp_out = Path(fp_out)

        for split in ["train", "valid", "test"]:
            for label in ["academic", "non_academic"]:
                (fp_out / split / label).mkdir(parents=True, exist_ok=True)

        for label in ["academic", "non_academic"]:
            label_folder = fp_in / label
            if not label_folder.exists():
                print(f"Warning: Folder {label_folder} does not exist, skipping...")
                continue

            imgs = list(label_folder.glob("*.jpeg"))
            np.random.shuffle(imgs)

            n_total = len(imgs)
            n_train = int(n_total * train_ratio)
            n_valid = int(n_total * valid_ratio)

            # Chunk it
            train_imgs = imgs[:n_train]
            valid_imgs = imgs[n_train:n_train+n_valid]
            test_imgs = imgs[n_train+n_valid:]

            for img in train_imgs:
                shutil.copy2(img, fp_out / "train" / label / img.name)
            for img in valid_imgs:
                shutil.copy2(img, fp_out / "valid" / label / img.name)
            for img in test_imgs:
                shutil.copy2(img, fp_out / "test" / label / img.name)

            print(f"{label}: Successfully split {n_total} images")
            print(f"Train: {len(train_imgs)} images")
            print(f"Valid: {len(valid_imgs)} images")
            print(f"Test: {len(test_imgs)} images")


if __name__ == "__main__":
    prep = Preprocessor()

    print("Preprocessing images...")

    print("\n[1/3] Processing academic set")
    prep.process_folder(
        fp_in = "../data/academic",
        fp_out = "../dataset-processed",
        label = "academic"
    )

    print("\n[2/3] Processing non-academic set")
    prep.process_folder(
        fp_in="../data/non_academic",
        fp_out="../dataset-processed",
        label="non_academic"
    )

    print("\n[3/3] Creating train-validation-test split")
    prep.split(
        fp_in="../dataset-processed",
        fp_out="../dataset-full",
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15
    )

    print("\nDone <3")