# *"An energy-efficient predictive model for Object Tracking Sensor Networks"* presented at the IEEE 5th World Forum on Internet of Things
### REQUIREMENTS
* *Python3*: on Linux `sudo apt-get install python3.6`;
* *scikit-learn*: `pip install -U scikit-learn`;
* *matplotlib*: `pip install -U matplotlib`;

### INSTRUCTIONS
1. Edit the *movements_predictions.ini* (if you wish);
2. Run the *mobile_object.py* file (`python3 mobile_object.py`);

## Files Description
### movements_predictions.ini
The simulator includes the following features:
- *max_area*: max_area where the mobile object moves into (m^2);
- *max_speed*: max speed of the mobile object (m/s);
- *deg_rand*: degree of movement randomness (0 <= *deg_rand* <= 1), where values closer to 0 refer to deterministic movements of the MO (it follows a certain pattern), whereas values closer to 1 refer to random movements;
- *n_steps*: number of steps/movements to simulate;
- *tot_predictions*: number of predictions the ML model has to perform.

### movements_simulator.py
Each coordinate has the following features:
- *pos* = position in Cartesian coordinate (0 up to max_area);
- *act* = action to perform (1=decrease, 2=stay, 3=increase);
- *speed* = speed of the MO (1 up to *max_speed*).

At each step/movement of the MO, a new location (*pos*) of the MO is chosen as random as the value of the *deg_rand* parameter according to the value of *act* and *speed*:
- if *act* is 1, the next location (*pos*) will be the difference between the current position and the value of *speed*;
- if *act* is 2, the next location (*pos*) will be the same of the current one, regardless of the value of *speed*;
- if *act* is 3, the next location (*pos*) will be the sum between the current position and the value of *speed*;

**NOTE:** in case the edge of the map is reached, the next location will be based on the steps to reach the edge plus/minus the remaining steps.

**example 1:** *max_area*=100, *pos*=98, *speed*=5, *act*=3 --> next pos=97;

**example 2:** *max_area*=100, *pos*=4, *speed*=10, *act*=1 --> next pos=6;

The effect of applying this procedure for both of the coordinates will result on the MO bouncing with the same angle against the map's edges.

### movements_predictor.py
Main steps:
1. The *train_set* will be based on the first 75% of the whole dataset and the *test_set* the remaining 25%;
2. A predefined range is chosen for *w_avg*, *w_ml* and *pol_deg*;
3. The automated algorithm:

	1. applies the cleaning procedure;
	2. applies the features and labels selection mechanism;
	3. computes new predictions;
	4. makes comparisons against the *test_set*;
	5. computes the predictions' accuracy
	6. compares the accuracy against the current best accuracy
	7. returns the best ML model (the one that produces the best accuracy)
	8. computes an amount of new predictions based on the number of *tot_predictions*.

### movements_predictor_test.py
Similar to **movements_predictor.py**, but step **3.vii** returns the result of the automated algorithm against the *test_set* without computing new predictions in order to see the actual results of the automated algorithm.

### mobile_object.py
Main file to execute that:
1. Defines a MO;
2. Generates simulation for each coordinate;
3. Either computes new predictions (test=True) or returns the result of the automated algorithm (test=False).
