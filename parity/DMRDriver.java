// Java driver: run Java MALLET's DMRTopicModel and dump topic-word weights and
// the learned DMR parameters (lambda), for statistical parity with rustmallet.DMR.
//
// Input file, one document per line:   feat1=val1,feat2=val2<TAB>tok tok ...
//   (an empty feature field is allowed; an intercept is added internally by DMR.)
// Outputs:
//   <out_counts>: MALLET's printTopicWordWeights  (topic \t word \t weight)
//   <out_params>: one line per topic: "<intercept> <f1> <f2> ..." plus a header
//                 line "#features intercept f1 f2 ..." giving feature order.
//
// Usage: DMRDriver <input> <numTopics> <iterations> <seed> <out_counts> <out_params>

import cc.mallet.classify.MaxEnt;
import cc.mallet.topics.DMRTopicModel;
import cc.mallet.types.Alphabet;
import cc.mallet.types.FeatureSequence;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class DMRDriver {
    public static void main(String[] args) throws Exception {
        String input = args[0];
        int numTopics = Integer.parseInt(args[1]);
        int iterations = Integer.parseInt(args[2]);
        int seed = Integer.parseInt(args[3]);
        String outCounts = args[4];
        String outParams = args[5];

        Alphabet dataAlphabet = new Alphabet();
        Alphabet featureAlphabet = new Alphabet();
        InstanceList instances = new InstanceList(dataAlphabet, featureAlphabet);

        BufferedReader br = new BufferedReader(new FileReader(input));
        String line;
        while ((line = br.readLine()) != null) {
            if (line.trim().isEmpty()) continue;
            String[] parts = line.split("\t", 2);
            String[] feats = parts[0].isEmpty() ? new String[0] : parts[0].split(",");
            String[] toks = parts.length > 1 ? parts[1].trim().split("\\s+") : new String[0];

            FeatureSequence fs = new FeatureSequence(dataAlphabet, Math.max(1, toks.length));
            for (String t : toks) fs.add(t);

            List<Integer> idx = new ArrayList<>();
            List<Double> vals = new ArrayList<>();
            for (String f : feats) {
                String[] kv = f.split("=");
                idx.add(featureAlphabet.lookupIndex(kv[0]));
                vals.add(Double.parseDouble(kv[1]));
            }
            int[] ii = new int[idx.size()];
            double[] vv = new double[idx.size()];
            for (int i = 0; i < idx.size(); i++) { ii[i] = idx.get(i); vv[i] = vals.get(i); }
            FeatureVector fv = new FeatureVector(featureAlphabet, ii, vv);

            instances.add(new Instance(fs, fv, null, null));
        }
        br.close();

        DMRTopicModel model = new DMRTopicModel(numTopics);
        model.setRandomSeed(seed);
        model.setNumIterations(iterations);
        model.addInstances(instances);
        model.estimate();

        model.printTopicWordWeights(new File(outCounts));

        MaxEnt p = model.getDmrParameters();
        double[] params = p.getParameters();
        int numFeatures = featureAlphabet.size() + 1;
        int defaultIndex = p.getDefaultFeatureIndex();

        PrintWriter pw = new PrintWriter(outParams);
        StringBuilder hdr = new StringBuilder("#features intercept");
        for (int f = 0; f < featureAlphabet.size(); f++) {
            hdr.append(" ").append(featureAlphabet.lookupObject(f).toString());
        }
        pw.println(hdr.toString());
        for (int topic = 0; topic < numTopics; topic++) {
            StringBuilder sb = new StringBuilder();
            sb.append(params[topic * numFeatures + defaultIndex]);
            for (int f = 0; f < featureAlphabet.size(); f++) {
                sb.append(" ").append(params[topic * numFeatures + f]);
            }
            pw.println(sb.toString());
        }
        pw.close();
    }
}
