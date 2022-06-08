import matplotlib.pyplot as plt

seq = [[40.97], [39.25, 43.81], [63.9, 93.74, 89.93]]
er = [[12.88], [12.96, 17.10], [43.62, 78.27, 33.45]]
ada = [[0.01], [0.04, 3.51], [0.67, 6.48, 0.89]]
frozen = [[0], [0, 0], [0, 0, 0]]


def avg(l):
    return sum(l)/len(l)

scores = {'Sequential Finetuning': seq,
          'Experience Replay': er,
          'Adapters': ada,
          'Frozen Encoder': frozen}

tasks = ['NLVR2', 'SNLI-VE', 'VCR']
#colors = ['gold', 'greenyellow', 'lightskyblue']
colors = ['darkorange', 'forestgreen', 'blue', 'red']
x = list(range(len(seq)))

for c, alg in zip(colors, scores.keys()):

    y = [avg(l) for l in scores[alg]]
    min_scores = [min(l) for l in scores[alg]]
    max_scores = [max(l) for l in scores[alg]]

    plt.plot(x, y, '-', color=c, label=alg)
    plt.plot(x, y, 'o', color=c)
    plt.fill_between(x, min_scores, max_scores, color=c, alpha=0.2)
xticks = ['Task {}: {}'.format(i+2, tasks[i]) for i in x]
plt.xticks(x, xticks)

plt.legend(loc='upper left')
plt.ylabel('Average Forgetting of Previous Tasks (%)')
plt.xlabel('Upstream Learning Task')
plt.show()