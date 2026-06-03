// Java driver: run Java MALLET's LabeledLDA on a labeled corpus and dump
// type-topic counts, for cross-implementation parity with rustmallet.LabeledLDA.
//
// Input file, one document per line:   label1,label2<TAB>tok tok tok ...
// Output file: first line = comma-separated label names in topic order;
//   then one line per word:   word  topic:count  topic:count ...
//
// Usage: LabeledLDADriver <input> <iterations> <seed> <alpha> <beta> <output>

import cc.mallet.topics.LabeledLDA;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class LabeledLDADriver {
    public static void main(String[] args) throws Exception {
        String input = args[0];
        int iterations = Integer.parseInt(args[1]);
        int seed = Integer.parseInt(args[2]);
        double alpha = Double.parseDouble(args[3]);
        double beta = Double.parseDouble(args[4]);
        String output = args[5];

        Alphabet dataAlphabet = new Alphabet();
        LabelAlphabet targetAlphabet = new LabelAlphabet();
        InstanceList instances = new InstanceList(dataAlphabet, targetAlphabet);

        BufferedReader br = new BufferedReader(new FileReader(input));
        String line;
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            String[] parts = line.split("\t", 2);
            String[] labs = parts[0].split(",");
            String[] toks = parts.length > 1 ? parts[1].trim().split("\\s+") : new String[0];

            FeatureSequence fs = new FeatureSequence(dataAlphabet, Math.max(1, toks.length));
            for (String t : toks) fs.add(t);

            List<Integer> idx = new ArrayList<>();
            for (String l : labs) {
                if (!l.isEmpty()) idx.add(targetAlphabet.lookupIndex(l));
            }
            int[] indices = new int[idx.size()];
            double[] values = new double[idx.size()];
            for (int i = 0; i < idx.size(); i++) { indices[i] = idx.get(i); values[i] = 1.0; }
            FeatureVector fv = new FeatureVector(targetAlphabet, indices, values);

            instances.add(new Instance(fs, fv, null, null));
        }
        br.close();

        LabeledLDA model = new LabeledLDA(alpha, beta);
        model.setRandomSeed(seed);
        model.setNumIterations(iterations);
        model.addInstances(instances);
        model.estimate();

        int[][] ttc = model.getTypeTopicCounts(); // [numTypes][numTopics]
        Alphabet vocab = model.getAlphabet();
        int numTopics = targetAlphabet.size();

        PrintWriter pw = new PrintWriter(output);
        StringBuilder hdr = new StringBuilder();
        for (int t = 0; t < numTopics; t++) {
            if (t > 0) hdr.append(",");
            hdr.append(targetAlphabet.lookupObject(t).toString());
        }
        pw.println(hdr.toString());
        for (int w = 0; w < vocab.size(); w++) {
            StringBuilder sb = new StringBuilder(vocab.lookupObject(w).toString());
            for (int t = 0; t < ttc[w].length; t++) {
                if (ttc[w][t] > 0) sb.append(" ").append(t).append(":").append(ttc[w][t]);
            }
            pw.println(sb.toString());
        }
        pw.close();
    }
}
