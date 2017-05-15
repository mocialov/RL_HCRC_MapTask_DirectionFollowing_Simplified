import random
from xml.dom import minidom
import math
import sys
import numpy
from scipy.spatial import distance
import cv2
import os

temperature = 2.0
a_lambda = 1.0

def savePolicyToFile(utterance, saving_to_file, a_string, others):
    with open("policy_"+saving_to_file+".txt", "ab+") as myfile:
        if others == 0:
            myfile.write("for utterance: "+utterance+" : "+a_string+"\n")
        else:
            myfile.write("for utterance: "+utterance+" : "+a_string+" but there are "+str(others)+" more\n")

def angleBetween(p1, p2):
    ang1 = numpy.arctan2(*p1[::-1])
    ang2 = numpy.arctan2(*p2[::-1])
    return numpy.rad2deg((ang1 - ang2) % (2 * numpy.pi))

def getNextDialogue(dialogue_transcript):
    instruction_giver_utterances = []
    with open('data/map0/transcript'+str(dialogue_transcript)+'.txt') as f:
        instruction_giver_utterances = f.readlines()
    return instruction_giver_utterances

#all landmarks
landmarks = ['m12_start', 'm12_caravan_park', 'm12_old_mill', 'm12_fenced_meadow', 'm12_abandoned_cottage', 'm12_fenced_meadow_2', 'm12_west_lake', 'm12_trig_point', 'm12_monument', 'm12_nuclear_test_site', 'm12_east_lake', 'm12_farmed_land', 'm12_finish']

xmldoc = minidom.parse('data/map0/landmarks.xml')

stops = []
itemlist = xmldoc.getElementsByTagName('stop')
for item in itemlist:
    stops.append((float(item.attributes['x'].value), float(item.attributes['y'].value)))

#all landmark locations
landmarks_locations = []
itemlist = xmldoc.getElementsByTagName('landmark')
for landmark in landmarks:
    for item in itemlist:
        if item.attributes['name'].value == landmark:
            for spot in item.getElementsByTagName('spot'):
                landmarks_locations.append((landmark, float(spot.attributes['x'].value), float(spot.attributes['y'].value)))



#all sides
sides = ['south', 'north', 'west', 'east']

#creating all possible actions
actions = []#[[None]]
for landmark in landmarks:
    #for side in sides:
    actions.append([landmark])

def createAllStates(instruction_giver_utterances):
    #creating all possible states
    states = []
    for utterance in instruction_giver_utterances:
        for landmark in landmarks:
            #for side in sides:
            states.append((landmark, utterance))
    return states

def createAllFeatures(states):
    #creating all possible features
    features = []
    for state in states:
        for action in actions:
            #coherence feature
            coherence_feature = 0
            for word in state[1].split():
                if(action[0] != None):
                    if(word in action[0]):
                        coherence_feature += 1
            #if (coherence_feature > 0):
            #    print "coherences found", coherence_feature
            #if (coherence_feature > 2):
            #    print "quite few", word, "in", state[1], coherence_feature, state[1].split(), "action", action[0]
    
            #landmark locality feature
            #locality_feature = False
            #if action[0] in shortest_paths and shortest_paths[action[0]] is state[0]: locality_feature = True
            #if state[0] in shortest_paths and shortest_paths[state[0]] is action[0]: locality_feature = True
    
            #if locality_feature:
            #    print state[0], " closest to ", action[0], locality_feature
    
            #sys.exit(0)
            #if(shortest[0]==action[0]):
            #    print "shortest between ", action[0], " and ", state[0]
            #landmark_locality_feature = "is" action[0] "closest to" state[1] "than to any other landmark?"
            #features.append( (state, action), () )

def transpose(an_array):
    return numpy.array(an_array).T

def pr(actions, utterance, current_action, state, theta, state_feature_dict):
    callee = "pr"
    selected_action = None
    selected_action_prob = numpy.array([0, 0, 0, 0, 0])
    for action in actions:
        a_sum = 0
        for action in actions:
            if action != current_action and action != None:
                a_sum += numpy.exp(\
                       numpy.array(make_feature_vector(callee, utterance, state, action)) * \
                       numpy.array(transpose(theta)) * \
                       1.0/temperature)

        probability = numpy.exp((1.0/temperature) * numpy.array(transpose(theta)) * numpy.array(make_feature_vector(callee, utterance, state, current_action))) / a_sum

        if(tuple(selected_action_prob) < tuple(probability)):
            selected_action = action
            selected_action_prob = probability

    return selected_action

#while True:
    #initializing
#    s_0 = states[random.randint(0, len(states)-1)]
    #a_0 = Pr(weights_vector, features_vector, t, states, actions)

def areAllWordsInString(utterance, searching_for):
    test_words = searching_for.split()
    contains_all = True

    for string in utterance:
        for test_word in test_words:
            if test_word not in string.split():
                contains_all = False
                break
        if not contains_all:
            break

    if contains_all:
        return True
    else:
        return False

def numberOfWords(searching_for_landmarks, utterance, resulting_landmark_after_the_action):
    number_of_words_matched = 0

    if searching_for_landmarks:
        #BOF filter
        if "2" in resulting_landmark_after_the_action:
            resulting_landmark_after_the_action = resulting_landmark_after_the_action.replace(" 2", "")
        #EOF filter
        
        #print "searching for: ", resulting_landmark_after_the_action, " in ", utterance
        
        if all(map(lambda w: w in utterance, tuple(resulting_landmark_after_the_action.split()))): return 1
        else: return 0
        #if all(x in utterance for x in resulting_landmark_after_the_action): 
        #    print "found"
        #    return 1
        #else: return 0

        '''for landmark in landmarks:
            landmark = landmark.replace("m12_","").replace("_", " ")
            if "2" in landmark:
                landmark = landmark.replace(" 2", "")
            #print "searching ", landmark, " in ", utterance, " and ", searching_for, " == ", landmark, (landmark in utterance) ," and ", (searching_for == landmark)
            if landmark in utterance:# and searching_for == landmark:
                #number_of_words_matched += 1
                print "FOUND!"
                return 1'''
    else:
        #print "raw utterance", utterance
        for landmark in landmarks:
            landmark = landmark.replace("m12_","").replace("_", " ")
            if "2" in landmark:
                landmark = landmark.replace("2", "")[:-1]
            utterance= utterance.replace(landmark,"")
        #print "filtered utterance", utterance
        for word in utterance.split():
            #print "searching for ", resulting_landmark_after_the_action, "in", word
            if resulting_landmark_after_the_action in word:
                #number_of_words_matched += 1
                #print "FOUND!"
                return 1


    #if number_of_words_matched > 0:
    #    print "searching for landmarks: ", searching_for_landmarks, " ||| looking for: ", resulting_landmark_after_the_action, " in ", utterance
    #    print "found: ", number_of_words_matched

    #print "searching for ", resulting_landmark_after_the_action, " in ", utterance
    #if resulting_landmark_after_the_action != None:
    #    if len(resulting_landmark_after_the_action.split()) > 1:
    #        if resulting_landmark_after_the_action in utterance:
    #            number_of_words_matched += 1
    #            #print "FOUND!"
    #    else:
    #        words = utterance.split()
    #        if resulting_landmark_after_the_action in words:
    #            number_of_words_matched += 1
    #            #print "FOUND!"
        
    return 0

def findShortestPathsBetweenLandmarks():
    #shortest path between landmarks
    shortest_paths = {}
    for landmark_location in landmarks_locations:
        shortest = (None, float("inf"))
        for landmark_location_2 in landmarks_locations:
            if(landmark_location != landmark_location_2):
                distance = math.sqrt( (landmark_location_2[1] - landmark_location[1])**2 + (landmark_location_2[2] - landmark_location[2])**2 )
                if distance < shortest[1]:
                    shortest = (landmark_location_2[0], distance)
        if (landmark_location[0] not in shortest_paths) or (shortest not in shortest_paths):
            shortest_paths[landmark_location[0]] = shortest[0]
    return shortest_paths

def make_feature_vector(callee, utterance, current_state, current_action):
    #print "callee", callee
    #coherence - number of words in utterance that occur in l' (current_action)

    coherence = 0
    if current_action[0] != None:
        coherence = numberOfWords(1, utterance, current_action[0].replace("m12_","").replace("_", " "))
    #print "words in utterance", utterance, " that occur in l' ", current_action[0], " is: ", coherence
    
    #landmark locality
    landmark_locality = 0
    if current_action[0] != None:
        if (shortest_paths_between_landmarks[current_action[0]] == current_state[0]) or (shortest_paths_between_landmarks[current_state[0]] == current_action[0]):
            #print "shortest path between", current_action[0], "and", current_state[0]
            landmark_locality = 1
    landmark_locality = 0
    
    #direction locality
    direction_locality = 0
    
    #null action
    null_action = int(current_action[0] == current_state[0])

    #Allocentric spatial
    #allocentric_spatial = 0

    #Egocentric spatial
    current_position = (-1, -1)
    target_position = (-1, -1)
    for landmark_location in landmarks_locations:
        if landmark_location[0] == current_state[0]:
            current_position_ = list(current_position)
            current_position_[0] = landmark_location[1]
            current_position_[1] = landmark_location[2]
            current_position = tuple(current_position_)
        if landmark_location[0] == current_action[0]:
            target_position_ = list(target_position)
            target_position_[0] = landmark_location[1]
            target_position_[1] = landmark_location[2]
            target_position = tuple(target_position_)

    angle = angleBetween(current_position, target_position)

    egocentric_spatial = 0
    '''if (angle >= 315.0 and angle < 45.0 and angle != 0.0): #above
        above = max(numberOfWords(0,utterance, "above"), numberOfWords(0,utterance, "north"))
        if (above > 0): egocentric_spatial = 1
    if (angle >= 135.0 and angle < 225.0): #below
        below = max(numberOfWords(0,utterance, "below"), numberOfWords(0,utterance, "south"))
        if (below > 0): egocentric_spatial = 1
    if (angle >= 45.0 and angle < 135.0): #right
        right = max(numberOfWords(0,utterance, "right"), numberOfWords(0,utterance, "east"))
        if (right > 0): egocentric_spatial = 1
    if (angle >= 225.0 and angle < 315.0): #left
        left = max(numberOfWords(0,utterance, "left"), numberOfWords(0,utterance, "west"))
        if (left > 0): egocentric_spatial = 1'''

    return [coherence*1.5, landmark_locality, direction_locality, null_action, egocentric_spatial]

def reward(utterance, current_state, next_state, current_action, next_action):
    callee = "reward"

    
    words = numberOfWords(1,utterance, current_action[0].replace("m12_","").replace("_", " "))

    

    
    current_state_index = landmarks.index(current_state[0])
    current_action_index = landmarks.index(current_action[0])
    #next_state_index = landmarks.index(next_state[0])
    #print current_state_index, next_state_index
    reward = 0.0

    #if (current_action[0] == current_state[0]) and (words == 0): reward = 1.0

    if ((current_state_index + 1) == current_action_index and words > 0):
        reward = 1.0
    #if (words > 0):
    #    print "reward 2 for doing ", current_action, "in state", current_state, "("+str(current_state_index)+") for utterance: ", utterance
    #    reward += 1.0
#    if reward == 2.0:
#       print "receiving reward: ", reward, "for doing ", next_action, "in state", current_state, " for utterance: ", utterance
    else:
        reward = -1.0
    

    #print "reward for doing ", current_action, "in state", current_state[0], " for utterance: ", utterance, " is: ", reward
    #print "words", words
    

    return reward #reward

#############################
shortest_paths_between_landmarks = findShortestPathsBetweenLandmarks()

theta = [random.uniform(0, 0.01),random.uniform(0, 0.01),random.uniform(0, 0.01),random.uniform(0, 0.01),random.uniform(0, 0.01)]

callee = "main"

iterations=100
random_action=True

saving_keyword = "iterations_" + str(iterations) + "_random_action_" + str(random_action)

state_feature_dict = {}

for i in range(1,iterations): #until theta converges
    for i in range(1, 8):
        instruction_giver_utterances = getNextDialogue(i)
        states = createAllStates(instruction_giver_utterances)
    
        #initialise
        current_state = (landmarks[0], instruction_giver_utterances[0], None)
        current_action = actions[random.randint(0, len(actions)-1)] #random first move temporary #pr(actions, current_action, current_state, theta, state_feature_dict)
    
        #print "begin theta: ", theta
    	
        for idx, utterance in enumerate(instruction_giver_utterances): #utterance or steps?
            if idx < len(instruction_giver_utterances)-1:
                #SARSA - I am in a state already, take action, new state and action

                #current feature
                feature_vector  = make_feature_vector(callee, utterance, current_state, current_action)
                
                #next state
                if current_action[0] != None:
                    next_state = (current_action[0], instruction_giver_utterances[idx+1], None)
                else:
                    next_state = (current_state[0], instruction_giver_utterances[idx+1], None)
                
                #next action
                next_action = actions[random.randint(0, len(actions)-1)] if random_action else pr(actions, utterance, current_action, current_state, theta, state_feature_dict)
                #print "next action for state", current_state[0], pr(actions, utterance, current_action, current_state, theta, state_feature_dict)
        
                #next feature
                next_feature_vector  = make_feature_vector(callee, instruction_giver_utterances[idx+1], next_state, next_action)
                
                #alpha
                alpha = 10.0/(10.0 + idx/i)

                #TAKE ACTION, GET REWARD
                if (current_action[0] != None):
                    #delta & theta
                    key = current_state[0] + "-" + str(current_action[0]) #'-'.join(item for item in list(current_state[0]) if item) + '-'.join(item for item in list(current_action[0]) if item)
                    if key in state_feature_dict:
                        #print reward(utterance, current_state, next_state, current_action, next_action), " + ", (numpy.array(transpose(state_feature_dict[key]))), " * ", numpy.array(next_feature_vector), " - ", (numpy.array(transpose(state_feature_dict[key]))), " * ", numpy.array(feature_vector)
                        delta = reward(utterance, current_state, next_state, current_action, next_action) + (numpy.array(transpose(state_feature_dict[key])) * numpy.array(next_feature_vector)) - (numpy.array(transpose(state_feature_dict[key])) * numpy.array(feature_vector))
                        state_feature_dict[key] = numpy.array(state_feature_dict[key]) + alpha * numpy.array(feature_vector) * numpy.array(delta) #map(sum, zip(state_feature_dict[key],(feature_vector * delta)))
                    else:
                        #print reward(utterance, current_state, next_state, current_action, next_action), " + ", (numpy.array(transpose(theta))), " * ", numpy.array(next_feature_vector), " - ", (numpy.array(transpose(theta))), " * ", numpy.array(feature_vector)
                        delta = reward(utterance, current_state, next_state, current_action, next_action) + (numpy.array(transpose(theta)) * numpy.array(next_feature_vector)) - (numpy.array(transpose(theta)) * numpy.array(feature_vector))
                        state_feature_dict[key] = numpy.array(theta) + alpha * numpy.array(feature_vector) * numpy.array(delta) #map(sum, zip(theta,(feature_vector * delta)))
        
                #update current state & action
                #NEW STATE AND ACTION
                current_state = next_state
                current_action = next_action

                #if(current_state[0] == "m12_finish"): continue
            else:
                continue

print len(state_feature_dict)
for item in state_feature_dict:
    print "end theta: ", item, state_feature_dict[item]

callee = "test"

draw_lines = []

#Testing
for i in range(8, 9):
    instruction_giver_utterances = getNextDialogue(i)
    states = createAllStates(instruction_giver_utterances)

    #initialise
    current_state = (landmarks[0], instruction_giver_utterances[0], None)

    for idx, utterance in enumerate(instruction_giver_utterances): #utterance or steps?
        if idx < len(instruction_giver_utterances)-1:
            smallest_Q = (-float('inf'),-float('inf'),-float('inf'),-float('inf'),-float('inf'))
            action_for_the_utterance = None
            for action in actions:
                #print "reward for action ", action, " in state", current_state, "has reward: ", reward(utterance, current_state, None, action, None)
                #current feature
                feature_vector  = make_feature_vector(callee, utterance, current_state, action)
                #print "feature: ", feature_vector, " for state ", current_state, " action ", action, " and utterance ", utterance
                key = current_state[0] + "-" + str(action[0])
                if key in state_feature_dict:
                    theta = state_feature_dict[key]
                    print "feature: ", feature_vector, " for state ", current_state, " action ", action, " and utterance ", utterance, " theta ", theta, " result: ", (numpy.array(theta) * numpy.array(feature_vector))
                    Q_state = numpy.array(theta) * numpy.array(feature_vector)
                    #print "comparing ", smallest_Q, " and ", tuple(Q_state), " - ", smallest_Q < tuple(Q_state)
                    if (smallest_Q < tuple(Q_state) and tuple(Q_state) != (0, 0, 0, 0, 0)):
                       smallest_Q = tuple(Q_state)
                       action_for_the_utterance = action
            #print "final Q", smallest_Q
            #is there theta with the same values in the dictionary? -> mean not fully explored and there are other routes
            others = 0
            #for action in actions:
            #    key = current_state[0] + "-" + str(action[0])
            #    if key in state_feature_dict:
            #        if tuple(state_feature_dict[key]) == smallest_Q:
            #            others += 1
            #print "for state: ", current_state[0], " take action ", action_for_the_utterance[0]
            savePolicyToFile(utterance, saving_keyword, "for state: "+ current_state[0]+ " take action "+ action_for_the_utterance[0], others)
            from_point = (-1, -1)
            to_point = (-1, -1)
            for landmark_location in landmarks_locations:
                if landmark_location[0] == current_state[0]:
                    from_point = (landmark_location[1], landmark_location[2])
                if landmark_location[0] == action_for_the_utterance[0]:
                    to_point = (landmark_location[1], landmark_location[2])
                line = [from_point, to_point]
            #print line
            draw_lines.append(line)
            current_state = (action_for_the_utterance[0], instruction_giver_utterances[idx+1], None)

            #if(current_state[0] == "m12_finish"): break


import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.lines import Line2D
fig, ax = plt.subplots()

fig = plt.figure(1, figsize=(10,10), dpi=20)
#fig.set_size_inches(10,10)          # Make graph square
#scatter([-0.1],[-0.1],s=0.01)     # Move graph window a little left and down

itemlist = xmldoc.getElementsByTagName('landmark')
for item in itemlist:
    for item2 in item.getElementsByTagName('spot'):
        plt.plot(float(item2.attributes['x'].value), float(item2.attributes['y'].value),'o', color='#f16824')
        ax = fig.add_subplot(111)

print "lines: ", len(draw_lines)
for idx, draw_line in enumerate(draw_lines):
    (line1_xs, line1_ys) = zip(*draw_line)
    ax.add_line(Line2D(line1_xs, line1_ys, linewidth=2, color='red'))

plt.plot()
#plt.show()
plt.savefig('trajectory'+saving_keyword+'.png')

def rotateImage(image, angle):
  image_center = tuple(numpy.array(image.shape)/2.0)
  rot_mat = cv2.getRotationMatrix2D(image_center,angle,0.5)
  result = cv2.warpAffine(image, rot_mat, image.shape,flags=cv2.INTER_LINEAR)
  return result

#rotate
img = cv2.imread('trajectory'+saving_keyword+'.png',0)
cv2.imwrite('trajectory_rotated'+saving_keyword+'.png', rotateImage(img, 180))

#flip
img = cv2.imread('trajectory_rotated'+saving_keyword+'.png',0)
rimg=cv2.flip(img,1)
cv2.imwrite('trajectory_flipped'+saving_keyword+'.png', rimg)

#remove temp files
os.remove('trajectory'+saving_keyword+'.png')
os.remove('trajectory_rotated'+saving_keyword+'.png')
os.rename('trajectory_flipped'+saving_keyword+'.png', 'trajectory_'+saving_keyword+'.png')