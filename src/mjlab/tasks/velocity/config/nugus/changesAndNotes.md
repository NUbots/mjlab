# 5795ea05f

Made the actor observations use gaussian noise with mean=0 and std=1. Along with a 5x increase in the action_scale (bitbots use 0.5 which is 10x), action distribution uses a log standard deviation now and the joint velocity noise is greatly reduced. 

Resulted in very noisy rewards and halved the overall mean reward value.

Maybe change the base_lin_vel observation back to uniform noise will fix this? Although in theory it shouldn't change anything since the actor pops this observation and the critic has enable_corruption=false. 

# 2dad58d63 

Reverted the base_lin_vel noise in observation vector. Resulted in no change (as expected). 

Thinking about the effects of the changes in 5795ea05f (the noise) I believe I made the standard deviations wayyy too large. So gonna fix that and should result in less noisy rewards and overall better tracking. 


