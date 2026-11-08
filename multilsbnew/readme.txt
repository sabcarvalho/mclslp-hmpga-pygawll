MULTILSB (Multi-Item Lot-Sizing with Backlogging) Library

Instructions on how to use:
===========================
1. Unzip the .zip file in any folder you want.

2. In the folder you unzipped the file, you can always reach the readme file 
   and the documentation provided.  Also, there is a mosel file that has the
   basic model of the problem (with data reading and parameter calculations)

3. There are 4 folders, one for each different test set.  When you open each 
   folder, there are two .txt files that contain the common data for that set.
   Also, there are 30 subfolders (for each instance of the set), and those
   contain the demand data.

4. When you run the mosel file, you need the following files for input:
   - periods.txt
   - multilsb_data.txt
   - demand.txt
   In the mosel file, lines 18-19 input the set and instance (you can automate this as
   input and hence be able to make a run with multiple instances if you like).
   The first two files are the same for all the instances of every set, and 
   demand data is different for each different instance.

5. The Excel file 'test_data.xls' provides all the LP relaxation bounds of the
   basic model, and heuristic solutions obtained using the heuristic in our up-
   coming paper. (Optimal solutions will be provided in near future for many
   of the instances.)

6. Feel free to use it and distribute it.

7. Please contact us if you have any problems or questions.
   Kerem Akartunali: Kerem.Akartunali@strath.ac.uk
   Andrew J. Miller: Andrew.Miller@math.u-bordeaux1.fr
