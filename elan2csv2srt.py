# -*- coding: utf-8 -*-

# Guidelines:
#   - Two lines of text on screen at most.
#   - 35-40 characters per line
#   - 160-180 words per minute
#   - minimum time of display to 1.5 seconds
#  SRT format:
#      1
#      00:00:12,000 --> 00:00:15,123
#      This is the first subtitle
#
#
#        <b></b> : bold
#        <i></i> : italic
#        <u></u> : underline
#        <font color=”#rrggbb”></font> : text color using 3 color components, red, green and blue.
import os
import numpy as np
import xml.etree.ElementTree as ElemTree

from config import KNOW_WORDS
from argparser import parse_arguments
from inference import evaluate_init_0
from inference import evaluate_bert


def init_outputs(out_path):
    if out_path:
        os.makedirs(out_path, exist_ok=True)
        out_path += "/" + file_eaf.split("/")[-1] if out_path[-1] != "/" else file_eaf.split("/")[-1]
        out_srt_name = out_path.replace('.eaf', '.srt')
        out_name = out_path.replace('.eaf', '.csv')
    else:
        out_srt_name = file_eaf.replace('.eaf', '.srt')
        out_name = file_eaf.replace('.eaf', '.csv')
    return out_srt_name, out_name


def read_conf(input_conf):
    with open(input_conf, 'r') as f:
        confidence = f.readlines()
    confidence = [i.split(" ")[3].strip() for i in confidence if len(i.split()) == 4]
    return np.asarray(confidence, dtype=float)


def read_eaf(input_eaf):
    tree = ElemTree.parse(input_eaf)
    root = tree.getroot()
    time_slot = root.findall("./TIME_ORDER/TIME_SLOT")

    ts_ms = {}
    for ts in time_slot:
        ts_ms[ts.get('TIME_SLOT_ID')] = ts.get('TIME_VALUE')

    segmentos = root.findall("./TIER[@TIER_ID='Word']/ANNOTATION/ALIGNABLE_ANNOTATION")
    if not segmentos:
        segmentos = root.findall("./TIER[@TIER_ID='Palabra']/ANNOTATION/ALIGNABLE_ANNOTATION")

    phrases = []
    for parent in segmentos:
        text = ''
        for child in parent:
            if child.text:
                text += child.text
                time_ini = int(ts_ms[parent.get('TIME_SLOT_REF1')])
                time_end = int(ts_ms[parent.get('TIME_SLOT_REF2')])
                phrases.append([time_ini, time_end, text])

    out_data = np.asarray(phrases)
    return out_data


def msec2time(msecs):
    secs, rmsecs = divmod(msecs, 1000)
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d.%03d' % (hours, mins, secs, rmsecs)


def msec2srttime(msecs):
    secs, rmsecs = divmod(msecs, 1000)
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d,%03d' % (hours, mins, secs, rmsecs)


def write_csv(input_csv, input_tini, input_tend, input_words):
    with open(input_csv, 'w', encoding='utf-8') as f:
        for t1, t2, word in zip(input_tini, input_tend, input_words):
            print("%s,%s,%.3f,%s,%.3f,%s,%.3f,%s" % ('Word', msec2time(t1), t1 / 1000, msec2time(t2), t2 / 1000,
                                                     msec2time(t2 - t1), (t2 - t1) / 1000, word), file=f)

        ind = 0
        t1 = input_tini[ind]
        t2 = input_tend[ind]
        phrase_text = input_words[0]

        while ind < len(input_tini) - 1:
            if input_tini[ind + 1] - input_tend[ind] < 250:
                phrase_text += ' ' + input_words[ind + 1]
                t2 = input_tend[ind + 1]
                ind += 1
            else:
                print("%s,%s,%.3f,%s,%.3f,%s,%.3f,%s" % ('Segment', msec2time(t1), t1 / 1000, msec2time(t2), t2 / 1000,
                                                         msec2time(t2 - t1), (t2 - t1) / 1000, phrase_text), file=f)
                print("%s,%s,%.3f,%s,%.3f,%s,%.3f,%s" % ('Speakers', msec2time(t1), t1 / 1000, msec2time(t2), t2 / 1000,
                                                         msec2time(t2 - t1), (t2 - t1) / 1000, '?'), file=f)

                phrase_text = input_words[ind + 1]
                t1 = input_tini[ind + 1]
                t2 = input_tend[ind + 1]
                ind += 1

        print("%s,%s,%.3f,%s,%.3f,%s,%.3f,%s " % ('Segment', msec2time(t1), t1 / 1000, msec2time(t2), t2 / 1000,
                                                  msec2time(t2 - t1), (t2 - t1) / 1000, phrase_text), file=f)
        print("%s,%s,%.3f,%s,%.3f,%s,%.3f,%s" % ('Speakers', msec2time(t1), t1 / 1000, msec2time(t2), t2 / 1000,
                                                 msec2time(t2 - t1), (t2 - t1) / 1000, ''), file=f)


def write_srt(input_srt, input_tini, input_tend, input_words, input_conf):

    def make_subs(i_words=input_words, i_tini=input_tini, i_tend=input_tend, i_conf=input_conf, p_conf=None):

        out_subtitles = {}
        phrase_text = i_words[0]
        p_confs = []
        phrase_confs = []

        idx = 0
        last_ind = 0
        t1 = i_tini[idx]
        t2 = i_tend[idx]
        word_count = 0
        srt_count = 1
        while idx < len(i_tini) - 1:
            if (i_tini[idx + 1] - i_tend[idx] < 800) and (word_count < 12) or len(phrase_text.split(" ")) < 6:

                new_word = i_words[idx + 1]
                if phrase_text.split(" ")[-1][-1] == ".":
                    new_word = new_word[0].upper() + new_word[1:]

                phrase_text += ' ' + new_word
                t2 = i_tend[idx + 1]
                idx += 1
                word_count += 1
            else:
                phrase_confs = list(i_conf[last_ind:idx + 1])
                if p_conf:
                    p_confs = list(p_conf[last_ind:idx + 1])
                    out_subtitles[srt_count] = (msec2srttime(t1), msec2srttime(t2), phrase_text, phrase_confs, p_confs)
                else:
                    out_subtitles[srt_count] = (msec2srttime(t1), msec2srttime(t2), phrase_text, phrase_confs)

                phrase_text = i_words[idx + 1]
                t1 = i_tini[idx + 1]
                t2 = i_tend[idx + 1]
                idx += 1
                last_ind = idx
                word_count = 1
                srt_count += 1
        if p_conf:
            out_subtitles[srt_count] = (msec2srttime(t1), msec2srttime(t2), phrase_text, phrase_confs, p_confs)
        else:
            out_subtitles[srt_count] = (msec2srttime(t1), msec2srttime(t2), phrase_text, phrase_confs)
        return out_subtitles

    def coloring_subs(subs):

        def conf2color(info_line):
            q = chr(34)
            yellow_font = '<font color=' + q + 'yellow' + q + '>'
            orange_font = '<font color=' + q + 'orange' + q + '>'
            red_font = '<font color=' + q + 'red' + q + '>'
            end_font = '</font>'

            def put_color(unc_word, x_conf):
                if x_conf > 0.9:
                    return unc_word
                elif x_conf > 0.7:
                    return yellow_font + unc_word + end_font
                elif x_conf > 0.5:
                    return orange_font + unc_word + end_font
                else:
                    return red_font + unc_word + end_font

            srt_word = info_line[2].split(" ")
            srt_conf = info_line[3]
            srt_conf_cap_punt = info_line[4]

            colered_line = ""
            for ind_word, word in enumerate(srt_word):
                if word.upper() in KNOW_WORDS:
                    # print(word, word[0].upper() + word[1:] + " ")
                    colered_line += word[0].upper() + word[1:] + " "
                elif word.isalnum() and word == word.lower():  # just asr confident to lower case
                    colered_line += put_color(word, srt_conf[ind_word]) + " "
                elif word == word.lower() and not word.isalpha():
                    colered_line += put_color(word[:-1], srt_conf[ind_word])
                    colered_line += put_color(word[-1], srt_conf_cap_punt[ind_word]) + " "
                elif word[0] == word[0].upper() and word != word.upper():
                    colered_line += put_color(word[0], srt_conf_cap_punt[ind_word])
                    if word.isalnum():
                        colered_line += put_color(word[1:], srt_conf[ind_word]) + " "
                    else:
                        colered_line += put_color(word[1:-1], srt_conf[ind_word])
                        colered_line += put_color(word[-1], srt_conf_cap_punt[ind_word]) + " "
                else:
                    colered_line += put_color(word, min(srt_conf[ind_word], srt_conf_cap_punt[ind_word]))

            return info_line[0], info_line[1], colered_line

        subs_c = subs.copy()
        for i in subs_c.keys():
            new_sub = conf2color(subs_c[i])
            subs[i] = new_sub

        return subs

    if args.pretrained_model == 'gtm-model':
        subtitles = make_subs()
        subtitles_punt_cap, subtitles_confidences = evaluate_init_0([sub[2] for sub in subtitles.values()])
        for ind, sub in enumerate(subtitles.values()):
            subtitles[ind + 1] = (sub[0], sub[1], subtitles_punt_cap[ind], sub[3], subtitles_confidences[ind])
    elif 'bert' in args.pretrained_model:
        words_punt_cap, punt_cap_confidences = evaluate_bert(list(input_words))
        subtitles = make_subs(i_words=words_punt_cap, p_conf=punt_cap_confidences)
    else:
        raise ValueError('Incorrect language argument for Dataset')
    color_subtitles = coloring_subs(subtitles)

    # Write down the colerd subs
    with open(input_srt, 'w', encoding='utf-8') as f:
        for key in color_subtitles:
            f.write("\n%d\n%s --> %s\n%s\n" % (key, color_subtitles[key][0], color_subtitles[key][1],
                                               color_subtitles[key][2]))
    return color_subtitles


if __name__ == '__main__':
    # load arguments
    args = parse_arguments()

    # initialization of file paths
    file_eaf = args.input
    conf_name = file_eaf.replace('.eaf', '.wordconfid.txt')
    srt_name, output_name = init_outputs(args.output)

    print('Leyendo datos ...')
    data = read_eaf(file_eaf)
    conf = read_conf(conf_name)
    tinit = data[:, 0].astype('int32')
    tend = data[:, 1].astype('int32')
    lines_asr = data[:, 2]

    print('Construyendo subtítulos ...')
    write_csv(output_name, tinit, tend, lines_asr)
    write_srt(srt_name, tinit, tend, lines_asr, conf)
    print('Subtítulos listos!')
