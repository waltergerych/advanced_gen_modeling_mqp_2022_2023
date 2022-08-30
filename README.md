## ``Advanced Applications of Deep Generative Models" MQP

**Advisor**: Elke Rundensteiner, **Mentor**: Walter Gerych

Incomplete, messy data is ubiquitous in real-world applications. The focus of this MQP is on generative methods to combat this issue and “fill in” incomplete data. Specifically, we will be focusing on developing novel training strategies for generative models in order to achieve this task. 

This is your MQP and provides an opportunity to show off your skills. Whether your goal is industry or academia, the MQP is useful for applications. The Advisor and Mentor serve as guides during the MQP. While we will provide support and a general direction, we want all of you to bring your own ideas and seek out new directions. 

**Acknowledgments**: This page borrows heavily from resource created by Prof. Tlachac. 

------------------------------------------

### Expectations

-	15 to 20 hours per week of work
-	Two formal meetings per week: one with advisor and one with mentor. Prepare slides for both meetings 
-	**At least** one peer meeting per week
-	Stay in communication with team (slack)
-	Teach team members as requested
-	Experience with being term leader and/or note taker. Leader ensures all team members are on time with term deliverables. Note taker records ideas for each meeting to share with team

### Deliverables

-	MQP report, improved each term
    - includes literature review
    - write one part, edit one part
    - cite all sources 
    - write succinctly (less can be more!)
    - include appendix with your term contributions
    - Potential to extend MQP report to full conference paper
-	Weekly update slides and presentation (for both advisor meeting and mentor meeting)
    - Include setup (don’t assume we remember!), what you did, and what you plan to do
    - Always include your conclusions and/or proposed next steps as appropriate
    - Complete slides and include written summary if unavailable to meet
    - **Send draft of advisor slides to mentor 1 day early**
-	Weekly meeting notes
-	Final presentation in D-term (poster or video as announced by the departments)
-	Term personal and team evaluations

### Evaluation

You will receive an individual grade. There are tangible and intangible contributions to the MQP team. You will be assessed on your ability to:
1.	define and meet project and personal goals
2.	problem solve and use feedback
3.	work independently and collaboratively
4.	communicate
5.	synthesize information and make conclusions

Further, we want you to demonstrate:
-	leadership and teamwork
-	academic integrity

### Important dates

- Advisor meetings
    - A term: 9am every Tuesday in Beckett conference room
- Mentor meetings
    - A term: 9am every Thursday in DS Innovation Lab (AK 013) 

### Resources 

Below you will find links to various resources that may aid in your MQP. We will periodically update this as the MQP progresses. 

**Relevant papers**

*GAN Papers*

- Original GAN paper: https://arxiv.org/pdf/1406.2661.pdf
    - See videos on GANs in videos section prior to reading
- Conditional GAN: Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets.": https://arxiv.org/pdf/1411.1784.pdf
    - Generating data with specific properties 
- Controllable GANs:  Lee, Minhyeok, and Junhee Seok. "Controllable generative adversarial network.": https://arxiv.org/pdf/1708.00598.pdf
    - An improvement over conditional GANs
- Shoshan, Alon, et al. "GAN-Control: Explicitly Controllable GANs.": https://arxiv.org/pdf/2101.02477.pdf
    - Another controllable GAN paper
- Geometric GAN: Lim, Jae Hyun, et al. "Geometric GAN": https://arxiv.org/pdf/1705.02894.pdf
    - A loss function based off of SVM properties
- Invertible Conditional GANs for image editing: Guim Perarnau, Joost van de Weijer, Bogdan Raducanu, and Jose M. Álvarez. https://arxiv.org/pdf/1611.06355.pdf
    - Allows the reconstruction and modifications of real face images conditioning on arbitrary attributes
- Literature Compilation and Review: Mescheder, Lars, et al. "Which Training Methods for GANs do actually Converge?": https://arxiv.org/pdf/1801.04406.pdf
    - Review of loss functions / training methods published in previous GAN research (2018 and earlier) 
- Tseng, P. "Convergence of a Block Coordinate Descent
Method for Nondifferentiable Minimization": https://link.springer.com/content/pdf/10.1023/A:1017501703105.pdf
    - Heavily cited paper of properties about gradient descent. 
- Bohm, Axel, et. al. "Two steps at a time — taking GAN training in stride
with Tseng’s method": https://arxiv.org/pdf/2006.09033v1.pdf
    - Quantifying the convergence of GAN training methods under specfic conditions.
- Jayagopal, Tarun Narain (2021) VICE-GAN: Video Identity-Consistent Emotion Generative Adversarial Network.: http://essay.utwente.nl/88376/1/Jayagopal_MA_EEMCS.pdf
- Progressive Growing GANs: https://arxiv.org/abs/1710.10196
    - Stable training of image-generating GANs by steadily increasing the sizes of the images generated
- Lin, Tianyi, et al. "On Gradient Descent Ascent for Nonconvex-Concave
Minimax Problems": https://arxiv.org/pdf/1906.00331.pdf
    - Proof based convergence Analysis
- Dauphin, Yann, et al. "Identifying and attacking the saddle point
problem in high-dimensional non-convex optimization": https://ganguli-gang.stanford.edu/pdf/14.SaddlePoint.NIPS.pdf
- Multi-Agent Diverse GANs: Arnab Ghosh, Viveka Kulharia, Vinay P. Namboodiri, Philip H.S. Torr, Puneet K. Dokania; Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 8513-8521 https://openaccess.thecvf.com/content_cvpr_2018/html/Ghosh_Multi-Agent_Diverse_Generative_CVPR_2018_paper.html

*Human Activity Recognition Papers*

- Vaizman, Yonatan, et al. "Context recognition in-the-wild: Unified model for multi-modal sensors and multi-label classification.": http://extrasensory.ucsd.edu/papers/vaizman2017b_imwutAcceptedVersion.pdf
- Jobanputra, Charmi, et al. "Human activity recognition: A survey.": https://reader.elsevier.com/reader/sd/pii/S1877050919310166?token=FDA56C10097234B41943474778C65C78108DB301165F04C9CCA19BC5CA757E0D9C0CEEBA0D7E946320D440BC651FF7D9&originRegion=us-east-1&originCreation=20220830161458

**Relevant blog posts**
- Jason Brownlee, "Deep Learning Models for Human Activity Recognition": https://machinelearningmastery.com/deep-learning-models-for-human-activity-recognition/
- Jason Brownlee, "A Gentle Introduction to a Standard Human Activity Recognition Problem": https://machinelearningmastery.com/how-to-load-and-explore-a-standard-human-activity-recognition-problem/
- Sylvain Combettes, "A basic intro to GANs (Generative Adversarial Networks)": https://towardsdatascience.com/a-basic-intro-to-gans-generative-adversarial-networks-c62acbcefff3
- Chapter 8. Conditional GAN: https://livebook.manning.com/book/gans-in-action/chapter-8/
- Bharath K, "Implementing Conditional Generative Adversarial Networks": https://blog.paperspace.com/conditional-generative-adversarial-networks/
    - This implementation is in Keras; read for theory not implementation 

**Relevant videos** 
- Basics of machine learning: https://www.youtube.com/watch?v=ukzFI9rgwfU
- Basics of GANs: https://www.youtube.com/watch?v=-Upj_VhjTBs
- Overview of original GAN paper
    - https://www.youtube.com/watch?v=8L11aMN5KY8
    - https://www.youtube.com/watch?v=Gib_kiXgnvA

**Tutorials**
- Jupyter Tutorials
    - https://www.youtube.com/watch?v=HW29067qVWk
- Machine learning in Python
    - https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
    - https://jakevdp.github.io/PythonDataScienceHandbook/05.02-introducing-scikit-learn.html
- Learning PyTorch
    - https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
- Building simple GAN in PyTorch
     - https://towardsdatascience.com/build-a-super-simple-gan-in-pytorch-54ba349920e4
- Building a conditional GAN in PyTorch: https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

**Tools** 
- Anaconda: https://www.anaconda.com/
    - Common Python distribution 
- Jupyter: https://jupyter.org/
    - For Deep Learning in Python
- Scikit-Learn: https://scikit-learn.org/stable/
    - Common Machine Learning package for Python
- Google scholar: https://scholar.google.com/
    - Great for literature reviews
    - Google Scholar button for quickly citing papers: https://chrome.google.com/webstore/detail/google-scholar-button/ldipcbpaocekfooobnbcddclnhejkcpn?hl=en

**Code Resources**
- PyTorch implementations of many types of GANs: https://cvnote.ddlee.cc/2019/09/25/gans-pytorch
- Another repo for GANs: https://github.com/w86763777/pytorch-gan-collections

**Datasets**
- ExtraSensory Dataset (in-the-wild HAR dataset): http://extrasensory.ucsd.edu/
- Scripted HAR Dataset: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones 

------------------------------------------

### Tasks
Below we will lists tasks. Every student that should perform the task will have their name as a sub bullet under the task. Students should ~~strikethrough~~ their names after they have completed the task. Students are expected to add to this list of tasks themselves. 

