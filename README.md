# TIL_TR1

## 1. [clean_dataset.py](https://github.com/HimanshuMittal01/ctr-optimization/blob/master/clean_dataset.py)

### Objective : Clean the dataset, drop rows, handle duplicates and missing values

### Implementation : Most simplest way i.e. to remove rows which have zero impressions

Remove rows having 0 impressions

## 2. [feature_selector.py](https://github.com/HimanshuMittal01/ctr-optimization/blob/master/feature_selector.py)

### Objective : Select and generate features from the cleaned dataset

### Implementation : Advertiser_ID, Month_and_year, Order_ID, sales_industry_id, Key_values, CTR

-   #### FS1

    #### Selected Encoding FS1

    -   Advertiser_ID : advertiser
    -   Month_and_year : month_and_year
    -   sales_industry_id : industry
    -   Key_values : key_values

    #### Dropped Values FS1

    -   Ad_server_targeted_clicks
    -   Ad_server_targeted_impressions
    -   Order_start_date
    -   Order_end_date
    -   Programmatic_order

-   #### FS2

    #### Month Label Encoding FS2

    -   May : 1
    -   April : 2
    -   June : 3
    -   July : 4
    -   October : 5
    -   Aug : 6
    -   November : 7
    -   February : 8
    -   September : 9
    -   January : 10
    -   March : 11
    -   December : 12

    #### Other encodings

    -   advertiser ONEHOT
    -   advertiser_count
    -   industry ONEHOT
    -   industries_count
    -   key_value
    -   months

## 3. [feature_transformation.py](https://github.com/HimanshuMittal01/ctr-optimization/blob/master/feature_transformation.py)

### Objective : Transform features for better predictions and prepare data for splits

### Implementation : No transformations, generated million-lines multilabel data

`random_combination` : A function to Generating random combinations.

`create_row` : A function to create a dict with all data and one hot vector of audience buckets of a single Order_ID depending on the feature set selected.

`time_probs`: A dictionary of probabilites of month_and_year based on count by dividing total occurences based on time by total data

`aggregate`: Create combinations of a particular Order_ID with the help of create_row

Next, we use lambdas to apply aggregate to the data grouped by Order_ID

## 4. [create_trajectories.py](https://github.com/HimanshuMittal01/ctr-optimization/blob/master/create_trajectories.py)

### Objective : Create trajectory given data state, action, and reward

### Implementation : Calculates propensity using softmax and create numpy array of form [s,a,r,p]

`create_single_trajectory`:A function to generate the following:

-   `context_matrix`: data, trivial -> `["advertiser","time_of_year","industry"]`
-   `action_matrix`: data, trivial, 1005 action buckets
-   `reward_vector`: data['ctr'].values \* REWARD_SIGN -> positive or negative depending on IPS or CRM
-   `propensity_matrix`: Softmax projection on action_matrix

We also tried logistic regression and SVM but they were not approximating the given distribution well enough.

## 5. [make_splits.py](https://github.com/HimanshuMittal01/ctr-optimization/blob/master/make_splits.py)

- Dev-F Shape      : (4, 11222)     3.2664 %
- Test-F Shape     : (4, 11225)     3.2673 %
- Dev-L Shape      : (4, 11196)     3.2588 %
- Test-L Shape     : (4, 10935)     3.1829 %
- Train-PO Shape   : (4, 254480)    74.0721 %
- Train-RP Shape   : (4, 44499)     12.9524 %

Train-PO, Train-RP, Dev-F, Test-F has same distribution, and dev-l, test-l consists of last month data and hance got different distribution
### Splitting Data

## 6. [experiment.py](https://github.com/HimanshuMittal01/ctr-optimization/blob/master/experiment.py)

### Objective: Policiy Network train

-   `REG_CONTROLLER`: Limiting layer

-   `crm_objective`: A funciton to calucate loss as (mean* + REG_CONTROLLER \* torch.sqrt(var* / BATCH_SIZE))

-   `calculate_baseline_loss` : A function to calculate baseline loss as follows:
```python
def calculate_baseline_loss(rewards,R,L):
    # R is variance regularizer
    # L is the length od the trajectory
    original_mean  = np.mean(rewards)
    original_reward_sq_sum = np.sum(np.square(rewards))
    original_var   = np.var(rewards)

    sqrtV = np.sqrt(original_var)
    sqrtL = np.sqrt(L)
    constant_a = 1.0-(R * original_mean)/ ((L-1)*sqrtV*sqrtL)
    constant_b = REG_CONTROLLER / (2*(L-1)*sqrtV*sqrtL)
    constant_c = (REG_CONTROLLER*sqrtV)/(2*sqrtL) + (REG_CONTROLLER*sqrtL*original_mean*original_mean)/(2*(L-1)*sqrtV)

    baseline = constant_a*original_mean + constant_b*original_reward_sq_sum + constant_c

    return original_mean, original_var, baseline
```
-   `policy_model` := PolicyNN1 a simple neural network with [32, 64, 64, 256] as hidden layers. Outputs the propensity

### Training loop:

Batchwise training to reduce the crm loss on the policy_model

### 7. train_rp.py

### Objecttive: Predict and analyze results from the policy_model and create a badit feedback

### Implementation : RandomForestRegressor

`estimator`: A sampling split to calculate RF2 and MAE and verify ctr prediction

### 8. [recommend_sets.py](https://github.com/HimanshuMittal01/ctr-optimization/blob/master/recommend_sets.py)

### Final Script. Takes input and gives output of recommended buckets.
