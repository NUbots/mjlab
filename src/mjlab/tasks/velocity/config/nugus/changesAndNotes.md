# 5795ea05f

Made the actor observations use gaussian noise with mean=0 and std=1. Along with a 5x increase in the action_scale (bitbots use 0.5 which is 10x), action distribution uses a log standard deviation now and the joint velocity noise is greatly reduced. 

Resulted in very noisy rewards and halved the overall mean reward value.

Maybe change the base_lin_vel observation back to uniform noise will fix this? Although in theory it shouldn't change anything since the actor pops this observation and the critic has enable_corruption=false. 

# 2dad58d63 

Reverted the base_lin_vel noise in observation vector. Resulted in no change (as expected). 

Thinking about the effects of the changes in 5795ea05f (the noise) I believe I made the standard deviations wayyy too large. So gonna fix that and should result in less noisy rewards and overall better tracking. 

# 4e085a9ce

Reduced the std devs of the noises on the sensors. Definitely improved the performance of the policy. When I resumed the training from a checkpoint the entropy curves did not jump sharply like they have for every other training run I've done. Not sure but retrained it without interrupting and it changed nothing, still weirdly different to everything else. Not necessarily an issue perhaps it means It's doing better. 

# 16678c60e

I've gone and measured the noise on the sensors and put them in as overrides for the nugus (hoepfully they apply like I expect). The sensors are quite smooth so I multiplied them by a large factor just to be safe. The numbers are still tiny so should improve things again. 


