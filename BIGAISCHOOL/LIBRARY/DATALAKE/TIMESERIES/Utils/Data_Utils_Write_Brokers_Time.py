from pathlib import Path
from os import fspath
def Write_Brokers_Times(Chart):
    data_path = Path(Path(
        __file__).resolve().parent.parent.parent) / "DATA\BROKERS_OPENCLOSE_TIMES.csv"
    data_path_last = fspath(data_path)
    #f = open(data_path_last, "w")
    #f.write(data[:])
    #f.close()
    oko=6