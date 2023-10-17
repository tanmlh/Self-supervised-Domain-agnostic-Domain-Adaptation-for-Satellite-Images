# Self-supervised-Domain-agnostic-Domain-Adaptation-for-Satellite-Images
1. Configure the dataset:
   Check how the dataset is organized in ./data folder.

3. Train an image2image network:

   python run.py configures/inria2sn2/dada_gen.py

4. Train the downstream network:

   python run.py configures/inria2sn2/dada.py

5. Test the performance of the downstream network:

   python run.py configures/inria2sn2/test_inria2sn2.py
