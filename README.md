# Notes

- Overview of work done:
    - Implementation of co-training wasn't successful; proposed method from the paper was too slow, while the classifiers were not really successful anyways.
    - Instead, MNIST dataset was used and first tested in *fixed-threshold* settings. Upper and lower threshold were found that were consistently giving +~5% accuracy compared to labeled dataset only. 
    - Adjusting the thresholds was then transformed into RL problem, with addition of states and rewards: 

        ```python
        R = Acc_t1 - Acc_t0
        R = R + 0.001 if R > 0 and len(selected_samples) > 0 else R
        ```

        ```python
        S = []
        for category in y:
            S += predicted probability distribution on *category* in *y*
        S += [len(selected_samples) / len(all_samples)]
        ```
    
    - PPO diverges to actions close to 0.0 or 1.0, which leads to selection of no samples or selection of all of them.
    - SAC doesn't learn at all and just gives out approx. random actions that coincidentally cause semi-successful training.

## Questions

- What would be the first thing that one would check in RL?
- Any suggestions for hyperparams?
- Multi-armed bandit problem?
- How to know that Markov property is not satisfied?

## Meeting notes:

- IMPLEMENT CO-TRAINING!
- PPO - lower LR, higher entropy weight, bigger model
- partial observability.

## TODO:

- Implement deeper NN.
- Try out different weight initialization. Check what are the outputs when the agent is initialized.