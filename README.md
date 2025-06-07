# Mimulus phenotyping pipeline

## Installation

1. **Clone the repository**:  
   Clone the repository to the local drive.
   ```
   git clone https://github.com/Salk-Harnessing-Plants-Initiative/mimulus-phenotyping.git
   ```

2. **Navigate to the cloned directory**:  
   
   ```
   cd mimulus-phenotyping
   ```

## Organize the pipeline and your trace files

Please make sure to organize the downloaded pipeline, and your trace files in the following architecture:

```
mimulus-phenotyping/
├── trace_pipeline/
│   ├── environment.yml
│   ├── pipeline.sh
│   ├── traits_trace.py
├── Traces/
│   ├── experimental design (e.g., A-06-19-24)/
│   │   ├── trace file (e.g., A-26-19-24-001.traces)
├── .gitignore
├── LICENSE
├── README.md
```

## Running the pipeline with a shell file 
1. **create the environment**:
   In terminal, navigate to the `trace_pipeline` folder and type:
   ```
   conda env create -f environment.yml
   ```
   or
   ```
   mamba env create -f environment.yml
   ```

2. **activate the environment**:
   ```
   conda activate mimulus-phenotyping
   ```

3. **run the shell file**:
   ```
   sh pipeline.sh
   ```

4. **check traits in csv files**:

   There will be a new folder (`traits_trace`) created after running the pipeline. Traits from all trace files can be found in `all_traits.csv` file. Meanwhile, there will be individual traits files for each experimental design (e.g., A-06-19-24).
