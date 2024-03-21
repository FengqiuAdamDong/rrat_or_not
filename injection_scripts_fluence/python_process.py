import psutil

def count_python_processes():
    python_processes = [p.info for p in psutil.process_iter(attrs=['name']) if 'python' in p.info['name'].lower()]
    return len(python_processes)

if __name__ == "__main__":
    print("Number of Python processes running:", count_python_processes())
