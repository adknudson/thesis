

# Motivating Data {#data}


It was Charles Darwin who in his book _On the Origin of Species_ developed the idea that living organisms adapt in order to better survive in their environment. Sir Francis Galton, inspired by Darwin's ideas, became interested in the differences in human beings and in how to measure those differences. Galton's works on studying and measuring human differences lead to the creation of psychometrics -- the science of measuring mental faculties. Around the same time that he was developing his theories, Johann Friedrich Herbart was also interested in studying consciousness through the scientific method, and is responsible for creating mathematical models of the mind.


E.H. Weber built upon Herbart's work, and sought out to prove the idea of a psychological threshold. A psychological threshold is a minimum stimulus intensity necessary to activate a sensory system -- a liminal stimulus. He paved the way for experimental psychology and is the namesake of Weber's Law -- the change in a stimulus that will be just noticeable is a constant ratio of the original stimulus [@britannica2014editors].


$$\frac{\Delta I}{I} = k$$


To put this law into practice, consider holding a 1 kg weight ($I = 1$), and further suppose that we can just detect the difference between a 1 kg weight and a 1.2 kg weight ($\Delta I = 0.2$). Then the constant just noticeable ratio is


$$k = \frac{0.2}{1} = 0.2$$


Now if we pick up a 10 kg weight, we should be able to determine how much more mass is required to just detect a difference:


$$\frac{\Delta I}{10} = 0.2 \Rightarrow \Delta I = 2$$


The difference between a 10 kg and a 12 kg weight should be just barely perceptible. Notice that the difference in the first set of weights is 0.2 and in the second set it is 2. The perception of the difference in stimulus intensities is not absolute, but relative. G.T. Fechner devised the law (Weber-Fechner Law) that the strength of a sensation grows as the logarithm of the stimulus intensity.


$$S = K \ln I$$


An example to this law is to consider two light sources, one that is 100 lumens ($S_1 = K \ln 100$) and another that is 200 lumens ($S_2 = K \ln 200$). The intensity of the second light is not perceived as twice as bright, but only about 1.15 times as bright according to the Weber-Fechner law:


$$\theta = S_2 / S_1 \approx 1.15$$


Notice that the value $K$ cancels out when calculating the relative intensity, but knowing $K$ can lead to important psychological insights; insights about differences between persons or groups of people. What biological and contextual factors affect how people perceive different stimuli? How do we measure their perception in a meaningful way? As one might expect, we can collect data from psychometric experiments, fit a model to the data from a family of functions called psychometric functions, and inspect key operating characteristics of those functions.


## Psychometric Experiments {#psycho-experiments}


Psychometric experiments are devised in a way to examine psychophysical processes, or the response between the world around us and our inward perceptions. A psychometric function relates an observer’s performance to an independent variable, usually some physical quantity of a stimulus in a psychophysical task [@wichmann2001a]. Psychometric functions were studied as early as the late 1800's, and Edwin Boring published a chart of the psychometric function in The American Journal of Psychology in 1917 [@boring1917chart].


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{figures/chart_of_pf} 

}

\caption{A chart of the psychometric function. The experiment in this paper places two points on a subject's skin separated by some distance, and has them answer their impression of whether there is one point or two, recorded as either 'two points' or 'not two points'. As the separation of aesthesiometer points increases, so too does the subject's confidence in their perception of 'two-ness'. So at what separation is the impression of two points liminal?}(\#fig:ch020-chart-of-pf)
\end{figure}


Figure \@ref(fig:ch020-chart-of-pf) displays the key aspects of the psychometric function. The most crucial part is the sigmoid function, the S-like non-decreasing curve which in this case is represented by the Normal CDF, $\Phi(\gamma)$. The horizontal axis represents the stimulus intensity -- the separation of two points in centimeters. The vertical axis represents the probability that a subject has the impression of two points. With only experimental data, the response proportion becomes an approximation for the probability.


This paper focuses on a type of psychometric experiment called a temporal order judgment (TOJ) experiment. If there are two distinct stimuli occurring nearly simultaneously then our brains will bind them into a single percept (perceive them as happening simultaneously). Compensation for small temporal differences is beneficial for coherent multisensory experiences, particularly in visual-speech synthesis as it is necessary to maintain an accurate representation of the sources of multisensory events. The temporal asynchrony between stimuli is called the stimulus onset asynchrony (SOA), and the range of SOAs for which sensory signals are integrated into a global percept is called the temporal binding window. When the SOA grows large enough, the brain segregates the two signals and the temporal order can be determined.


Our experiences in life as we age shape the mechanisms of processing multisensory signals, and some multisensory signals are integrated much more readily than others. Perceptual synchrony has been previously studied through the point of subjective simultaneity (PSS) -- the temporal delay between two signals at which an observer is unsure about their temporal order [@stone2001now]. The temporal binding window is the time span over which sensory signals arising from different modalities appear integrated into a global percept. 


A deficit in temporal sensitivity may lead to a widening of the temporal binding window and reduce the ability to segregate unrelated sensory signals. In TOJ tasks, the ability to discriminate the timing of multiple sensory signals is referred to as temporal sensitivity, and is studied through the measurement of the just noticeable difference (JND) -- the smallest lapse in time so that a temporal order can just be determined. 


Figure \@ref(fig:ch020-plot-ref-pf) highlights the features through which we study psychometric functions. The PSS is defined as the point where an observer can do no better at determining temporal order than random guessing (i.e.  when the response probability is 50%). The JND is defined as the extra temporal delay between stimuli so that the temporal order is just able to be determined. Historically this has been defined as the difference between the 84% level -- one standard deviation away from the mean -- and the PSS, though the upper level often depends on domain expertise.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{020-psychometrics_files/figure-latex/ch020-plot-ref-pf-1} 

}

\caption{The PSS is defined as the point where an observer can do no better at determining temporal order than random guessing. The just noticeable difference is defined as the extra temporal delay between stimuli so that the temporal order is just able to be determined. Historically this has been defined as the difference between the 0.84 level and the PSS, though the upper level depends on domain expertise.}(\#fig:ch020-plot-ref-pf)
\end{figure}


Perceptual synchrony and temporal sensitivity can be modified through a baseline understanding. In order to perceive physical events as simultaneous, our brains must adjust for differences in temporal delays of transmission of both psychical signals and sensory processing [@fujisaki2004recalibration]. In some cases such as with audiovisual stimuli, the perception of simultaneity can be modified by repeatedly presenting the audiovisual stimuli at fixed time separations (called an adapter stimulus) to an observer [@vroomen2004recalibration]. This repetition of presenting the adapter stimulus is called temporal recalibration.


## Temporal Order Judgment Task {#toj-task}


The data set used in this paper comes from experiments done by A.N. Scurry and Dr. F. Jiang in the Department of Psychology at the University of Nevada. Reduced temporal sensitivity in the aging population manifests in an impaired ability to perceive synchronous events as simultaneous, and similarly more difficulty in segregating asynchronous sensory signals that belong to different sources. The consequences of a widening of the temporal binding window is considered in @scurry2019aging, as well as a complete detailing of the experimental setup and recording process. A shortened summary of the methods is provided below.


There are four different tasks in the experiment: audio-visual, visual-visual, visual-motor, and duration, and each task is respectively referred to as audiovisual, visual, sensorimotor, and duration. The participants consist of 15 young adults (age 20-27), 15 middle age adults (age 39-50), and 15 older adults (age 65-75), all recruited from the University of Nevada, Reno. Additionally all subjects are right handed and were reported to have normal or corrected to normal hearing and vision.


\begin{table}[!h]

\caption{(\#tab:ch020-multitask-data)Sample of motivating data.}
\centering
\begin{tabular}[t]{rrllllrl}
\toprule
soa & response & sid & task & trial & age\_group & age & sex\\
\midrule
-350 & 0 & O-m-BC & audiovisual & pre & older\_adult & 70 & M\\
-200 & 0 & M-m-SJ & duration & post1 & middle\_age & 48 & M\\
28 & 1 & O-f-KK & sensorimotor & pre & older\_adult & 66 & F\\
275 & 1 & O-f-MW & visual & post1 & older\_adult & 69 & F\\
\bottomrule
\end{tabular}
\end{table}


In the audiovisual TOJ task, participants were asked to determine the temporal order between an auditory and visual stimulus. Stimulus onset asynchrony values were selected uniformly between -500 to +500 ms with 50 ms steps, where negative SOAs indicated that the visual stimulus was leading, and positive values indicated that the auditory stimulus was leading. Each SOA value was presented 5 times in random order in the initial block. At the end of each trial the subject was asked to report if the auditory stimulus came before the visual, where a $1$ indicates that they perceived the sound first, and a $0$ indicates that they perceived the visual stimulus first.


A similar setup is repeated for the visual, sensorimotor, and duration tasks. The visual task presented two visual stimuli on the left and right side of a display with temporal asynchronies that varied between -300 ms to +300 ms with 25 ms steps. Negative SOAs indicated that the left stimulus was first, and positive that the right came first. A positive response indicates that the subject perceived the right stimulus first.


The sensorimotor task has subjects focus on a black cross on a screen. When it disappears, they respond by pressing a button. Additionally, when the cross disappears, a visual stimulus was flashed on the screen, and subjects were asked if they perceived the visual stimulus before or after their button press. The latency of the visual stimulus was partially determined by individual subject's average response time, so SOA values are not fixed between subjects and trials. A positive response indicates that the visual stimulus was perceived after the button press.


The duration task presents two vertically stacked circles on a screen with one appearing right after the other. The top stimulus appeared for a fixed amount of time of 300 ms, and the bottom was displayed for anywhere between +100 ms to +500 ms in 50 ms steps corresponding to SOA values between -200 ms to +200 ms. The subject then responds to if they perceived the bottom circle as appearing longer than the top circle.


\begin{table}[!h]

\caption{(\#tab:ch020-toj-summary)Summary of TOJ Tasks}
\centering
\begin{tabular}[t]{lll}
\toprule
Task & Positive Response & Positive SOA Truth\\
\midrule
Audiovisual & Perceived audio first & Audio came before visual\\
Visual & Perceived right first & Right came before left\\
Sensorimotor & Perceived visual first & Visual came before tactile\\
Duration & Perceived bottom as longer & Bottom lasted longer than top\\
\bottomrule
\end{tabular}
\end{table}


After the first block of each task was completed, the participants went through an adaptation period where they were presented with the respective stimuli from each task repeatedly at fixed temporal delays, then they repeated the task. To ensure that the adaptation affect persisted, the subject were presented with the adapter stimulus at regular intervals throughout the second block. The blocks are designated as `pre` and `post1`, `post2`, etc. in the data set. In this paper we only focus on the `pre` and `post1` blocks.


## Data Visualizations and Quirks


The dependent variable in these experiments is the perceived response which is encoded as a 0 or a 1, and the independent variable is the SOA value. If the response is plotted against the SOA values, then it is difficult to determine any relationship (see figure \@ref(fig:ch020-simple-response-soa-plot)). Transparency can be used to better visualize the relationships between SOA values and responses. The center plot in figure \@ref(fig:ch020-simple-response-soa-plot) uses the same data as the left plot, except that the transparency is set to 0.05. Note that there is a higher density of "0" responses towards more negative SOAs, and a higher density of "1" responses for more positive SOAs. The proportion of "positive" responses for a given SOA may be computed and plotted against the SOA value. This is displayed in the right panel. Now the relationship between SOA values and responses is clear -- as the SOA value goes from more negative to more positive, the proportion of positive responses increases from near 0 to near 1.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{020-psychometrics_files/figure-latex/ch020-simple-response-soa-plot-1} 

}

\caption{Left: Simple plot of response vs. soa value. Center: A plot of response vs. soa with transparency. Right: A plot of proportions vs. soa with transparency.}(\#fig:ch020-simple-response-soa-plot)
\end{figure}


Subjectively the right plot in figure \@ref(fig:ch020-simple-response-soa-plot) is the easiest to interpret. Because of this, we will often present the observed and predicted data using the proportion of responses rather than the actual response. Proportional data also has the advantage of being bounded on the same interval as the response in contrast to the raw counts.


For the audiovisual task, the responses can be aggregated into binomial data -- the number of positive responses for given SOA value -- which is sometimes more efficient to work with than the Bernoulli data (see table \@ref(tab:ch020-av-bin-sample)). However the number of times an SOA is presented varies between the pre-adaptation and post-adaptation blocks; 5 and 3 times per SOA respectively. 


\begin{table}[!h]

\caption{(\#tab:ch020-av-bin-sample)Audiovisual task with aggregated responses.}
\centering
\begin{tabular}[t]{lrrrr}
\toprule
trial & soa & n & k & proportion\\
\midrule
 & 200 & 5 & 4 & 0.80\\
\cmidrule{2-5}
 & 150 & 5 & 5 & 1.00\\
\cmidrule{2-5}
\multirow[t]{-3}{*}{\raggedright\arraybackslash pre} & -350 & 5 & 0 & 0.00\\
\cmidrule{1-5}
 & 350 & 3 & 3 & 1.00\\
\cmidrule{2-5}
 & -500 & 3 & 1 & 0.33\\
\cmidrule{2-5}
\multirow[t]{-3}{*}{\raggedright\arraybackslash post1} & -200 & 3 & 0 & 0.00\\
\bottomrule
\end{tabular}
\end{table}


Other quirks about the data pertain to the subjects. There is one younger subject that did not complete the audiovisual task, and one younger subject that did not complete the duration task. Additionally there is one older subject who's response data for the post-adaptation audiovisual task is unreasonable -- it is extremely unlikely that the data represents genuine responses (see figure \@ref(fig:ch020-av-post1-O-f-CE-plot)).


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{020-psychometrics_files/figure-latex/ch020-av-post1-O-f-CE-plot-1} 

}

\caption{Post-adaptation response data for O-f-CE}(\#fig:ch020-av-post1-O-f-CE-plot)
\end{figure}


It is unreasonable because, of all the negative SOAs, there were only two "correct" responses (the perceived order matches the actual order). If a subject is randomly guessing the temporal order, then a naive estimate for the proportion of correct responses is 0.5. If a subject's proportion of correct responses is above 0.5, then they are doing better than random guessing. Figure \@ref(fig:ch020-av-post-neg-trials) shows that subject O-f-CE is the only one who's proportion is below 0.5 (and by a considerable amount).


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{020-psychometrics_files/figure-latex/ch020-av-post-neg-trials-1} 

}

\caption{Proportion of correct responses for negative SOA values during the post-adaptation audiovisual experiment.}(\#fig:ch020-av-post-neg-trials)
\end{figure}


When this method of detecting outliers is repeated for all tasks and blocks, then we end up with 17 records in total (figure \@ref(fig:ch020-naive-prop-outliers)), one of which is the aforementioned subject.


\begin{figure}

{\centering \includegraphics[width=0.85\linewidth]{020-psychometrics_files/figure-latex/ch020-naive-prop-outliers-1} 

}

\caption{Proportion of correct responses across all tasks and blocks Proportions are calculated individually for positive and negative SOAs.}(\#fig:ch020-naive-prop-outliers)
\end{figure}


Most of the records that are flagged by this method of outlier detection are from the sensorimotor task, and none are from the visual task. This may be attributed to the perceived difficulty of the task. One consequence of higher temporal sensitivity is that it is easier to determine temporal order. It may also be that determining temporal order is inherently easier for certain multisensory tasks compared to others. Since the sensorimotor task does not have fixed SOA values like the other tasks, it may be perceived as more difficult. Or perhaps the mechanisms that process tactile and visual signals are not as well coupled as those that process audio and visual signals.
