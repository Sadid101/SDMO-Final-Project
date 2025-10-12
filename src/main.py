from src import config

def main():
    print("Application started.")
    if config.DEBUG:
        print("Debug ENV working")
    print("Ready for coding!")

if __name__ == "__main__":
    main()