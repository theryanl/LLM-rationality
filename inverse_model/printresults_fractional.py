'''
Print out a set of results from the decision events card sorting
experiment
'''

# Take a card number and convert to a decision event
def cardnum2event(num):
    if num == 24:
        return 'dc/b/ax'
    elif num == 3:
        return 'ab/ax'
    elif num == 46:
        return 'bad/bac/bax'
    elif num == 0:
        return 'a/x'
    elif num == 1:
        return 'dcb/ax'
    elif num == 20:
        return 'b/a/x'
    elif num == 17:
        return 'd/c/bax'
    elif num == 21:
        return 'dc/b/a/x'
    elif num == 26:
        return 'c/b/ax'
    elif num == 5:
        return 'dcb/a/x'
    elif num == 13:
        return 'bax'
    elif num == 18:
        return 'x'
    elif num == 34:
        return 'b/ax'
    elif num == 23:
        return 'ac/bx/ax'
    elif num == 15:
        return 'cb/ax'
    elif num == 19:
        return 'dc/bax'
    elif num == 33:
        return 'd/c/b/ax'
    elif num == 42:
        return 'ad/ac/ab/ax'
    elif num == 4:
        return 'dcba/x'
    elif num == 22:
        return 'ad/cx/bx/ax'
    elif num == 14:
        return 'bdx/bcx/bax'
    elif num == 25:
        return 'ax'
    elif num == 6:
        return 'dcx/bax'
    elif num == 27:
        return 'cbax/dbax'
    elif num == 36:
        return 'cba/x'
    elif num == 32:
        return 'cx/bx/ax'
    elif num == 29:
        return 'dc/bx/ax'
    elif num == 39:
        return 'dcbax'
    elif num == 12:
        return 'cb/a/x'
    elif num == 38:
        return 'cbax'
    elif num == 8:
        return 'bad/bcx/bax'
    elif num == 7:
        return 'bdx/cax/bax'
    elif num == 37:
        return 'cax/bax'
    elif num == 10:
        return 'd/c/b/a/x'
    elif num == 41:
        return 'bac/bax'
    elif num == 16:
        return 'ad/ac/bx/ax'
    elif num == 11:
        return 'bdc/bax'
    elif num == 45:
        return 'd/cbax'
    elif num == 28:
        return 'ac/ab/ax'
    elif num == 35:
        return 'c/bax'
    elif num == 44:
        return 'c/b/a/x'
    elif num == 30:
        return 'ba/x'
    elif num == 40:
        return 'cbad/cbax'
    elif num == 31:
        return 'bx/ax'
    elif num == 2:
        return 'dx/cx/bx/ax'
    elif num == 9:
        return 'dc/ba/x'
    elif num == 43:
        return 'dc/ab/ax'
    else:
        print 'Invalid input!\n'
        
# Take a string formatted like abc/def/ghi and reformat it like
# abc
# def
# -----
# ghi
def formatevent(e):
    options = e.split('/')
    eNew = ''
    for o in options:
        eNew += o
        eNew += '\n'
    return eNew


# Take a list of events and format them so they appear on the same row
def formateventlist(elist):
    formattedElist = ''
    # First separate by newlines
    elistNew = []
    for e in elist:
        elistNew.append(e.split('\n'))
    # Loop through the rows (max 5)
    for i in range(5):
        # Append the first rows of each event with space in between
        wroteRow = 0
        for e in elistNew:
            # How long is this row
            if len(e)-1 < i:
                nChars = 0
                option = ''
            else:
                nChars = len(e[i])
                option = e[i]
                wroteRow = 1
            # Compute number of additional spaces needed
            nSpaces = 9 - nChars
            # Now print out the row and the spaces
            formattedElist += option
            for s in range(nSpaces):
                formattedElist += ' '
        # Add a newline
        if wroteRow == 1:
            formattedElist += '\n'
    return formattedElist

#Calculate mean and standard deviation of data x[]: 
#mean = {\sum_i x_i \over n} std = sqrt(\sum_i (x_i - mean)^2 \over n-1)
def meanstdv(x): 
    from math import sqrt 
    n, mean, std = len(x), 0, 0 
    for a in x: 
        mean = mean + a 
    mean = mean / float(n) 
    for a in x: 
        std = std + (a - mean)**2 
    std = sqrt(std / float(n-1)) 
    return mean, std 

# Compare two floating point numbers
def cmpEventMeans(x1,x2):
    if x1[1] < x2[1]:
        return -1
    elif x1[1] > x2[1]:
        return 1
    else:
        if x1[2] < x2[2]:
            return -1
        elif x1[2] > x2[2]:
            return 1
        else:
            return 0
            
# Read in the numbers and look for errors in file
# f is an open file descriptor
def checkFile(f):
    cardNums = []
    for line in f:
        # Strip off the newlines
        # Separate the cards that were put on the same line
        cards = line.strip().split(',')
        # Convert cards to ints and sort
        for c in cards:
            cardNums.append(int(c))
    cardNums.sort()
    if (min(cardNums) != 0 or max(cardNums) != 46):
        print 'Invalid card range'
        return -1
    for i in range(len(cardNums)):
        if (cardNums[i] != i):
            print 'Missing or double card'
            print cardNums
            return -1
    
    return 1
        
    

#################################################################
# Main routine
#################################################################

    
import sys

resultFiles = sys.argv[1:]

# Open a results output file
fout = open('fullresults.txt','w')
# Create an array to record sorting positions of all the cards
sortingPositions = list()
for i in range(47):
    sortingPositions.append([]);
    # 2d list, first dim is 47 (num cards), second dim is for each participant

for fname in resultFiles: # each file that is passed in
    fout.write('======== ' + fname + ' ========\n\n')
    fout.write('Weakest\n\n')
    # Open each raw data file to read
    f = open(fname)

    # ensures file has one copy of each number 0-46
    if (checkFile(f) == -1):
        print 'Bad file: ' + fname
    
    # Read through one line at a time
    f = open(fname)
    position = 1
    for line in f:
        # Strip off the newlines
        # Separate the cards that were put on the same line
        cards = line.strip().split(',')

        # Convert cards to ints and sort
        cardNums = [] # list of card numbers
        for c in cards:
            cardNums.append(int(c))
        cardNums.sort() # sort the cards that are tied in this line

        events = []
        fractionalRanking = sum(range(position, position+len(cardNums))) / float(len(cardNums))
        # average ranking of position to position + number tied

        for n in cardNums: # the index of the outer loop of sortingpositions, 
            # i.e., the columns in the table rawdata_fractional.csv are literally the card numbers
            # cardnum2event maps it to the actual decision


            # Record the fractional position of each card number
            sortingPositions[n].append(fractionalRanking)
            # Convert each card number to a decision event
            ev = cardnum2event(n)
            # Nicely format as an event
            events.append(formatevent(ev))
        # Write out a row of events
        fout.write(formateventlist(events) + '\n')
        #print(formateventlist(events))

        position += len(cardNums)
    fout.write('Strongest\n\n\n')
fout.close()    

# Print out the raw data in an csv format
fouttable = open('rawdata_fractional.csv','w')
for i in range(len(resultFiles)): # outer loop: participant
    for j in range(len(sortingPositions)): # inner loop: card number: range(47)
        fouttable.write(str(sortingPositions[j][i]) + ',')
    fouttable.write('\n')


means = []
for i in range(len(sortingPositions)):
    m,sd = meanstdv(sortingPositions[i])
    means.append([i, m, sd])
    
    

means.sort(cmpEventMeans)
fout2 = open('averageresults.txt','w')
for n in means:
    ev = cardnum2event(n[0])
    event = formatevent(ev)
    fout2.write(event + ' ' + str(n[1]) + ' ' + str(n[2]) + '\n')
fout2.close()

# Output matlab-friendly files
labelout = open('labels_fractional.csv','w')
meansout = open('means_fractional.csv','w')
stdsout = open('stds_fractional.csv','w')
for n in means:
    ev = cardnum2event(n[0])
    labelout.write(ev + '\n')
    meansout.write(str(n[1]) + '\n')
    stdsout.write(str(n[2]) + '\n')
labelout.close()
meansout.close()
stdsout.close()


