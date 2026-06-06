from pathlib import Path


def main():
    print("Dataset: ISIC 2019 Training Input and GroundTruth CSV")
    print("Canonical source: https://challenge.isic-archive.com/data/#2019")
    print("Expected layout after download:")
    print("  ISIC_2019_Training_Input/")
    print("  ISIC_2019_Training_Input/ISIC_2019_Training_GroundTruth.csv")
    print()
    print("This dataset may require accepting the ISIC archive terms before download.")
    print(f"Current data directory exists: {Path('ISIC_2019_Training_Input').exists()}")


if __name__ == "__main__":
    main()
