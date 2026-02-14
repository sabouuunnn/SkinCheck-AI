import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Image, ScrollView, ActivityIndicator, Dimensions } from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import * as FileSystem from 'expo-file-system/legacy'; 
import * as Location from 'expo-location';
import * as tf from '@tensorflow/tfjs';
import { decodeJpeg } from '@tensorflow/tfjs-react-native';
import { Ionicons } from '@expo/vector-icons';

const { width } = Dimensions.get('window');
const MODEL_URL = 'https://teachablemachine.withgoogle.com/models/tC3cWVP4h/';

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState('Загрузка...');
  const [isModelReady, setIsModelReady] = useState(false);
  const [model, setModel] = useState<any>(null);
  const [labels, setLabels] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [weather, setWeather] = useState<any>(null);

  useEffect(() => {
    (async () => {
      try {
        let { status } = await Location.requestForegroundPermissionsAsync();
        if (status === 'granted') {
          let loc = await Location.getCurrentPositionAsync({});
          const res = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${loc.coords.latitude}&longitude=${loc.coords.longitude}&current=temperature_2m,uv_index&timezone=auto`);
          const data = await res.json();
          setWeather(data.current);
        }

        await tf.ready();
        const loadedModel = await tf.loadLayersModel(MODEL_URL + 'model.json');
        const metaRes = await fetch(MODEL_URL + 'metadata.json');
        const meta = await metaRes.json();
        
        setLabels(meta.labels);
        setModel(loadedModel);
        setIsModelReady(true);
        setResult('Система готова');
      } catch (e) {
        setResult('Ошибка запуска');
        console.log("ОШИБКА ПРИ СТАРТЕ:", e);
      }
    })();
  }, []);

  const pickImage = async () => {
    let res = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'], 
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.5,
    });

    if (!res.canceled && res.assets) {
      const uri = res.assets[0].uri;
      setImage(uri);
      analyze(uri);
    }
  };

  const analyze = async (uri: string) => {
    if (!model) return;
    setLoading(true);
    setResult('ИИ думает...');

    try {
      const base64Data = await FileSystem.readAsStringAsync(uri, {
        encoding: 'base64', 
      });
      
      const uint8Array = tf.util.encodeString(base64Data, 'base64').buffer;
      const raw = new Uint8Array(uint8Array);
      const imageTensor = decodeJpeg(raw);

      const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
      const expanded = resized.expandDims(0);
      const normalized = expanded.div(127.5).sub(1);

      const prediction = model.predict(normalized) as any;
      const data = await prediction.data();
      
      const maxIndex = data.indexOf(Math.max(...data));
      const confidence = (data[maxIndex] * 100).toFixed(1);
      const diagnosis = labels[maxIndex] || "Анализ завершен";

      setResult(`${diagnosis}\nТочность: ${confidence}%`);

      tf.dispose([imageTensor, resized, expanded, normalized, prediction]);
    } catch (e) {
      console.log("ОШИБКА АНАЛИЗА:", e);
      setResult('Ошибка: Не удалось распознать');
    } finally {
      setLoading(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.header}>SkinCheck AI</Text>

      {weather && (
        <View style={styles.weatherCard}>
          <Text style={styles.weatherTxt}>🌡 {weather.temperature_2m}°C</Text>
          <Text style={styles.weatherTxt}>☀️ UV: {weather.uv_index}</Text>
          <Text style={[styles.advice, {color: weather.uv_index > 3 ? 'red' : 'green'}]}>
            {weather.uv_index > 3 ? "Нужен SPF!" : "Безопасно"}
          </Text>
        </View>
      )}

      <View style={styles.card}>
        <View style={styles.imgBox}>
          {image ? <Image source={{ uri: image }} style={styles.img} /> : <Ionicons name="camera-outline" size={60} color="#ccc" />}
        </View>

        <View style={styles.resBox}>
          {loading ? <ActivityIndicator size="large" color="#007AFF" /> : <Text style={styles.resTxt}>{result}</Text>}
        </View>

        <TouchableOpacity 
          style={[styles.btn, !isModelReady && {backgroundColor:'#ccc'}]} 
          onPress={pickImage} 
          disabled={!isModelReady}
        >
          <Text style={styles.btnText}>{isModelReady ? "ВЫБРАТЬ СНИМОК" : "ЗАГРУЗКА..."}</Text>
        </TouchableOpacity>
      </View>
      
      <Text style={styles.footer}>Проект для учебных целей.</Text>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { padding: 20, alignItems: 'center', backgroundColor: '#F2F2F7', minHeight: '100%', paddingTop: 60 },
  header: { fontSize: 32, fontWeight: 'bold', marginBottom: 20 },
  weatherCard: { flexDirection: 'row', backgroundColor: '#fff', padding: 15, borderRadius: 20, marginBottom: 20, width: '100%', justifyContent: 'space-around', alignItems: 'center' },
  weatherTxt: { fontSize: 16, fontWeight: '600' },
  advice: { fontWeight: 'bold' },
  card: { backgroundColor: '#fff', padding: 20, borderRadius: 30, width: '100%', alignItems: 'center' },
  imgBox: { width: width - 80, height: width - 80, backgroundColor: '#f8f9fa', borderRadius: 20, justifyContent: 'center', alignItems: 'center', overflow: 'hidden' },
  img: { width: '100%', height: '100%' },
  resBox: { marginVertical: 20 },
  resTxt: { fontSize: 18, fontWeight: '700', textAlign: 'center' },
  btn: { backgroundColor: '#007AFF', padding: 18, borderRadius: 15, width: '100%', alignItems: 'center' },
  btnText: { color: '#fff', fontWeight: 'bold', fontSize: 16 },
  footer: { marginTop: 25, fontSize: 11, color: '#8e8e93', textAlign: 'center' }
});