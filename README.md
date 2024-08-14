# BC baseline

File Dir Structure

``````
├─bc_train.py
│  
├─bc_models
│  ├─cramped_room
│  │      
│  └─forced_coordination
│          
├─bc_results
│  ├─cramped_room         
│  └─forced_coordination
│              
├─overcooked_ai_py
│          
└─overcooked_expert_data
    ├─cramped_room   
    └─forced_coordination

``````

You can run this code using 

``````python
python BCEval/bc_train.py
``````

And the run result will be saved in bc_results, and the trained model will be saved in bc_models.



# MARLEval

``````

├─logger.py
├─recoder.py
├─replay_buffer.py
├─requirements.txt
├─test_marl.py
├─train_offline.py
├─train_offline2.py
├─train_online.py
├─ train_online2.py
│  
├─config
│          
├─experiment
│  └─2024.08.05
│      ├─cramped_room_eval 
│      │  ├─tb     
│      │  └─video         
│      ├─forced_coordination_eval
│      │  ├─tb 
│      │  └─video     
│      ├─maddpg_cramped_offline-vanilla    
│      │  ├─tb  
│      │  └─video        
│      ├─maddpg_cramped_online_vanilla  
│      │  ├─buffer     
│      │  ├─tb 
│      │  └─video      
│      ├─maddpg_forced_offline-vanilla
│      │  ├─tb    
│      │  └─video      
│      └─maddpg_forced_online_vanilla
│          ├─buffer       
│          ├─tb   
│          └─video
├─model
│  └─utils
├─overcooked_ai_py
│              
├─result
│  │ 
│  └─data
│          
├─test
│  │  
│  └─buffer
│          
└─utils
``````

The online training data is saved in folders maddpg_forced_online_vanilla/buffer and maddpg_cramped_online_vanilla/buffer in our experiment, in both layout cramped_room and forced_coordination, 0.25M interactions were performed.

You can collect more data by following the command if you want.

``````
python MARLEval/train_online.py
python MARLEval/train_online2.py
``````

And then, you need to run the following command for offline trainning.

``````
python MARLEval/train_offline.py
python MARLEval/train_offline2.py
``````

You can then test it by running the following command.

```
python MALREval/test_marl.py
```

The result will be saved in folder {layout_name}_eval/video.

