import numpy as np
import math as ma
import sys
import time


def cum_gain(relevance):
    """
    Calculate cumulative gain.
    This ignores the position of a result, but may still be generally useful.
    @param relevance: Graded relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    return np.asarray(relevance).sum()


def dcg(relevance, alternate=True):
    """
    Calculate discounted cumulative gain.
    @param relevance: Graded and ordered relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    @param alternate: True to use the alternate scoring (intended to
    place more emphasis on relevant results).
    @type alternate: C{bool}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    rel = np.asarray(relevance)
    p = len(rel)

    if alternate:
        # from wikipedia: "An alternative formulation of
        # DCG[5] places stronger emphasis on retrieving relevant documents"

        log2i = np.log2(np.asarray(range(1, p + 1)) + 1)
        return ((np.power(2, rel) - 1) / log2i).sum()
    else:
        log2i = np.log2(range(2, p + 1))
        return rel[0] + (rel[1:] / log2i).sum()


def idcg(relevance, alternate=True):
    """
    Calculate ideal discounted cumulative gain (maximum possible DCG).
    @param relevance: Graded and ordered relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    @param alternate: True to use the alternate scoring (intended to
    place more emphasis on relevant results).
    @type alternate: C{bool}
    """

    if relevance is None or len(relevance) < 1:
        return 0.0

    # guard copy before sort
    rel = np.asarray(relevance).copy()
    rel.sort()
    return dcg(rel[::-1], alternate)


def ndcg(relevance, nranks, alternate=True):
    """
    Calculate normalized discounted cumulative gain.
    @param relevance: Graded and ordered relevances of the results.
    @type relevance: C{seq} or C{numpy.array}
    @param nranks: Number of ranks to use when calculating NDCG.
    Will be used to rightpad with zeros if len(relevance) is less
    than nranks
    @type nranks: C{int}
    @param alternate: True to use the alternate scoring (intended to
    place more emphasis on relevant results).
    @type alternate: C{bool}
    """
    if relevance is None or len(relevance) < 1:
        return 0.0

    if (nranks < 1):
        raise Exception('nranks < 1')

    rel = np.asarray(relevance)
    pad = max(0, nranks - len(rel))

    # pad could be zero in which case this will no-op
    rel = np.pad(rel, (0, pad), 'constant')

    # now slice downto nranks
    rel = rel[0:min(nranks, len(rel))]

    ideal_dcg = idcg(rel, alternate)
    if ideal_dcg == 0:
        return 0.0

    return dcg(rel, alternate) / ideal_dcg
class OfflineEnv(object):

    def __init__(self, users_dict, users_history_lens, movies_id_to_movies, state_size, fix_user_id=None):

        self.users_dict = users_dict
        self.users_history_lens = users_history_lens
        self.items_id_to_name = movies_id_to_movies

        self.state_size = state_size
        self.available_users = self._generate_available_users()

        self.fix_user_id = fix_user_id

        self.user = fix_user_id if fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items = set(self.items)
        self.done_count = 3000

    def _generate_available_users(self):
        available_users = []
        for i, length in zip(self.users_dict.keys(), self.users_history_lens):
            if length > self.state_size:
                available_users.append(i)
        return available_users

    def reset(self):
        self.user = self.fix_user_id if self.fix_user_id else np.random.choice(self.available_users)
        self.user_items = {data[0]: data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        self.recommended_items =set(self.items)
        return self.user, self.items, self.done

    def step(self, action, k=False):
        i=0
        x=np.zeros(39967)

        #print(action)


        for e,v in self.user_items.items():
            if v>0:
                if e>0 and e<396:
                    i=0
                    x[i]=x[i]+(1/len(self.user_items))
                if e>395 and e<791:
                    i=1
                    x[i]=x[i]+(1/len(self.user_items))
                if e>790 and e<1186:
                    i=2
                    x[i]=x[i]+(1/len(self.user_items))
                if e>1185 and e<1581:
                    i=3
                    x[i]=x[i]+(1/len(self.user_items))
                if e>1580 and e<1976:
                    i=4
                    x[i]=x[i]+(1/len(self.user_items))
                if e>1975 and e<2371:
                    i=5
                    x[i]=x[i]+(1/len(self.user_items))
                if e>2370 and e<2766:
                    i=6
                    x[i]=x[i]+(1/len(self.user_items))
                if e>2765 and e<3161:
                    i=7
                    x[i]=x[i]+(1/len(self.user_items))
                if e>3160 and e<3556:
                    i=8
                    x[i]=x[i]+(1/len(self.user_items))
                if e>3555 and e<3953:
                    i=9
                    x[i]=x[i]+(1/len(self.user_items))

        # reward = -0.5
        if k:
            correctly_recommended = []
            rewards = []
            # for act in action:
            #     if act in self.user_items.keys() and act not in self.recommended_items:
            #         correctly_recommended.append(act)
            #         rewards.append((self.user_items[act] - 3) / 2)
            #     else:
            #         rewards.append(-0.5)
            #     self.recommended_items.add(act)
            # if max(rewards) > 0:
            #     self.items = self.items[len(correctly_recommended):] + correctly_recommended
            # reward = rewards

        else:
            reward=-0.001

            if action in self.user_items.keys() and action not in self.recommended_items:
                if self.user_items[action]>0:
                    if action > 0 and action < 396:
                        i = 0

                    if action > 395 and action < 791:
                        i = 1

                    if action > 790 and action < 1186:
                        i = 2

                    if action > 1185 and action < 1581:
                        i = 3

                    if action > 1580 and action < 1976:
                        i = 4

                    if action > 1975 and action < 2371:
                        i = 5

                    if action > 2370 and action < 2766:
                        i = 6

                    if action > 2765 and action < 3161:
                        i = 7

                    if action > 3160 and action < 3556:
                        i = 8

                    if action > 3555 and action < 3953:
                        i = 9

                    reward = (1/10) - x[i]+1  # reward
                else:
                    reward = -0.001

            if reward > 0:
                self.items = self.items[1:] + [action]
            self.recommended_items.add(action)

        cvr = 0
        propfair = 0


        if len(self.recommended_items) > self.done_count or len(self.recommended_items) >= self.users_history_lens[
            self.user - 1]:
            self.done = True
            pp = np.zeros(len(self.recommended_items))
            for re in self.recommended_items:
                    ttt=0
                    for d, q in self.user_items.items():
                        if re == d:
                           print(str(q))
                           cvr = cvr + (1 / (len(self.recommended_items)))
                           if q>=0 and q<=1:
                               pp[ttt]=0
                           if q>1 and q<=2:
                               pp[ttt]=1
                           if q>2 and q<=3:
                               pp[ttt]=2
                           if q>3 and q<=5:
                               pp[ttt]=3
                    ttt=ttt+1

            for it in x:
                propfair=propfair+ma.log(1+it)

            print("--------------------------------------------------------")
            ufg=propfair/(1-cvr)
            if ufg > 6.00:
                print(pp)
                print(f'cvr : {cvr:0.3f},  propfair : {propfair:0.3f}, ufg : {ufg:0.3f}, NDCG : {ndcg(pp,len(pp)):0.3f}''\n')
                print("Run Time: " + str(time.time()))
                sys.exit(0)

        return self.items, reward, self.done, self.recommended_items

    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_names
