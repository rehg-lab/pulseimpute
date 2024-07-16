from utils import visualize


def main():
    visualize(task='PTBXL', ptbxl1='Extended', ptbxl2='10%', models=['FFT_ptbxl'],
                                                                     sample_index=0, x_range=(0.0, 9999.0))
    
if __name__ == "__main__":
    main()